# SPDX-License-Identifier: AGPL-3.0-or-later
# sovmr — build & launch convenience. `make` (or `make run`) is all you need.

CARGO      ?= cargo
BIN        := target/release/sovmr
PREFIX     ?= $(HOME)/.local
BINDIR     := $(PREFIX)/bin
DESKTOPDIR := $(PREFIX)/share/applications
ICONDIR    := $(PREFIX)/share/icons/hicolor/scalable/apps
APPDIR     := dist/sovmr.AppDir
APPIMAGE   := dist/sovmr-x86_64.AppImage

# appimagetool is fetched once into dist/tools/ (no root, no system
# install) and runs in extract mode so FUSE is never required.
APPIMAGETOOL     := dist/tools/appimagetool-x86_64.AppImage
APPIMAGETOOL_URL := https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage

.PHONY: run mock report build test lint verify reproducible sha256sums e2e e2e-windows windows-ffmpeg test-long test-mkv test-audio windows windows-check install uninstall deploy-windows versions package clean

## Default: optimized build + launch the GUI.
run: build
	./$(BIN)

## Launch in simulation mode (no GPU needed) — UI verification.
mock: build
	./$(BIN) --mock

## Headless hardware check.
report: build
	./$(BIN) --report

build:
	$(CARGO) build --release -p sovmr-app

## Workspace unit + doc tests.
test:
	$(CARGO) test --workspace

## Static quality gate: clippy (warnings are errors) + formatting check.
lint:
	$(CARGO) clippy --workspace --all-targets -- -D warnings
	$(CARGO) fmt --all --check

## The full offline gate in one command: static analysis, the whole
## unit/doc suite, then the Windows cross-check — the exact chain every
## change goes through. Run by hand a dozen times before it earned a
## target: a repetitive command chain is a script, not muscle memory.
## Prerequisites run left to right and make stops at the
## first failure, so a red gate is reported, never walked past.
##
## Deliberately excludes the live gates (`e2e`, `e2e-windows`, `test-mkv`,
## `test-audio`, `test-long`): those need a GPU, the other machine, or
## ffmpeg as a fixture generator, none of which a plain `verify` may
## assume present. They are named here so the omission is a decision, not
## an oversight — run them explicitly where their inputs exist.
verify: lint test windows-check
	@echo "verify: static gate + workspace unit/doc tests + Windows cross-check all passed"

## Reproducible-build gate: the same source
## must produce the same bytes, so anyone can rebuild the release and
## confirm the published binary is exactly this code with nothing slipped
## in. Measured, never assumed — it builds the release binary, forces a
## full re-codegen and relink of the final crate, builds again, and fails
## loudly if the two differ. It holds today because the identity carries
## no build timestamp (see `build/identity.rs`) and `codegen-units = 1`
## makes codegen deterministic; this target is what keeps it holding.
##
## Kept out of `verify`: two fat-LTO release builds are minutes, and the
## everyday gate must stay quick.
reproducible: build
	@cp $(BIN) $(BIN).repro1
	$(CARGO) clean --release -p sovmr-app
	$(CARGO) build --release -p sovmr-app
	@if cmp -s $(BIN) $(BIN).repro1; then \
		echo "reproducible: two independent builds are byte-identical ($$(sha256sum $(BIN) | cut -c1-16)…)"; \
		rm -f $(BIN).repro1; \
	else \
		echo "NOT REPRODUCIBLE: $$(cmp -l $(BIN) $(BIN).repro1 | wc -l) bytes differ — the release cannot be verified by rebuild"; \
		rm -f $(BIN).repro1; \
		exit 1; \
	fi

## Publish-ready SHA-256 of the release artifacts. With a
## reproducible build (`make reproducible`), a published hash lets anyone
## confirm a downloaded binary is byte-for-byte this source — the plain,
## universal half of proving a release is genuine, no secret in it. Hashes
## every artifact that is actually built; fails loudly when there is
## nothing to hash rather than writing an empty, falsely reassuring file.
sha256sums:
	@present=""; \
	for f in $(BIN) $(WINDOWS_OUT)/sovmr.exe $(WINDOWS_OUT)/sovmr-setup.exe $(APPIMAGE); do \
		[ -f "$$f" ] && present="$$present $$f"; \
	done; \
	if [ -z "$$present" ]; then \
		echo "sha256sums: nothing built — run 'make build', 'make windows', or 'make package' first"; \
		exit 1; \
	fi; \
	mkdir -p dist; \
	sha256sum $$present | tee dist/SHA256SUMS

## Real-GPU end-to-end encode test (needs NVDEC+NVENC; ffmpeg is used
## only to generate the input fixture).
##
## Serial on purpose. NVENC allows a bounded number of concurrent
## sessions (8 on this consumer driver), and cargo's default parallel
## harness opens one per test plus eight more inside the cancellation
## gate — past the cap the driver refuses `nvEncOpenEncodeSessionEx`
## with code 21 and a perfectly good test reports a failure it did not
## cause. The contention is in the harness, not the engine.
e2e:
	$(CARGO) test -p sovmr-native --test e2e_pipeline -- --ignored --test-threads=1

## Matroska reader against a file our own code did not write.
## Needs ffmpeg to make the fixture, so it is opt-in like the GPU gates:
## the unit tests prove the reader agrees with itself, this proves it
## agrees with somebody else's muxer.
test-mkv:
	$(CARGO) test -p sovmr-mkv --test real_file -- --ignored --nocapture
	$(CARGO) test -p sovmr-mkv --test write_read -- --ignored --nocapture

## Lossless audio decoders against ffmpeg, sample for sample. A lossless
## decoder is either exact or broken, so this is the only gate that
## means anything for it.
test-audio:
	$(CARGO) test -p sovmr-audio --test bit_exact -- --ignored --nocapture
	$(CARGO) test -p sovmr-audio --test oracle -- --ignored --nocapture
	$(CARGO) test -p sovmr-native --test audio_route -- --ignored --nocapture

## Cross-compile the Windows program and its installer.
##
## The linker is mingw-w64 GCC, named in .cargo/config.toml. The
## installer carries the program inside itself via include_bytes!, so it
## must be built second, with SOVMR_PAYLOAD pointing at the freshly built
## exe — hence two invocations rather than one.
WINDOWS_TARGET := x86_64-pc-windows-gnu
WINDOWS_OUT := target/$(WINDOWS_TARGET)/release

## The Windows test machine: an ssh alias, so the address lives in
## ~/.ssh/config rather than in this file.
WINHOST ?= win-ci
## Every call to it is bounded: a network call without a hard
## timeout is a way for the build to hang forever when the other machine
## is simply switched off — which is its ordinary state, not a fault.
SSH := ssh -o BatchMode=yes -o ConnectTimeout=10
SCP := scp -o BatchMode=yes -o ConnectTimeout=10
## Static Windows ffmpeg, for `windows-ffmpeg`. A test fixture generator,
## the same role ffmpeg plays on Linux — never linked, never shipped.
FFMPEG_WIN_URL ?= https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip
## Where work lands on it. A directory it owns, not Program Files.
WINWORK ?= C:/sovmr-ci

windows:
	$(CARGO) build --release --target $(WINDOWS_TARGET) -p sovmr-app
	SOVMR_PAYLOAD="$(CURDIR)/$(WINDOWS_OUT)/sovmr.exe" \
	  $(CARGO) build --release --target $(WINDOWS_TARGET) -p sovmr-setup
	@echo "windows build ready:"
	@ls -lh $(WINDOWS_OUT)/sovmr.exe $(WINDOWS_OUT)/sovmr-setup.exe

## Everything compiles for Windows, including the layout asserts that
## catch MSVC/GCC padding differences at build time.
##
## `--all-targets`, so tests are checked too. Without it a test using a
## Unix-only import compiled here and nowhere else — which is exactly
## the kind of gap a cross-check exists to close.
windows-check:
	$(CARGO) check --workspace --all-targets --target $(WINDOWS_TARGET)

## The same live gates, on the other machine and the other GPU
## generation. Until this existed the Ampere branch had never run: every
## live gate was Linux/Ada, and an RTX 3060 encodes HEVC where a 4070
## encodes AV1 — different code, zero coverage.
##
## The test binary is cross-built here and shipped whole; nothing is
## compiled there. `ffmpeg.exe` lives beside it in $(WINWORK) rather than
## being installed into the system, because it is a fixture generator
## and not part of what ships.
##
## Serial for the same reason as `e2e`: NVENC caps concurrent sessions.
e2e-windows:
	@test -x "$$(command -v ssh)" || { echo "ssh is required"; exit 1; }
	@mkdir -p $(WINDOWS_OUT)
	# Build on its own line so a cross-compile failure is THIS recipe's
	# failure. Piped straight into `sed | tail` the pipeline's exit status
	# was the tail's, not cargo's, so a broken cross-build reported success
	# and the gate went green on nothing — the exact "a step that failed
	# did not say so" this project forbids. The JSON goes to a file; cargo's
	# own error diagnostics still reach the terminal on stderr (no longer
	# swallowed by `2>/dev/null`).
	$(CARGO) test --no-run --target $(WINDOWS_TARGET) -p sovmr-native \
		--test e2e_pipeline --message-format=json > $(WINDOWS_OUT)/.e2e-json
	sed -n 's/.*"executable":"\([^"]*e2e_pipeline[^"]*\)".*/\1/p' \
		$(WINDOWS_OUT)/.e2e-json | tail -1 > $(WINDOWS_OUT)/.e2e-path
	@test -s $(WINDOWS_OUT)/.e2e-path || { echo "cross-build produced no test binary"; exit 1; }
	$(SSH) $(WINHOST) "if not exist \"$(subst /,\\,$(WINWORK))\" mkdir \"$(subst /,\\,$(WINWORK))\""
	$(SCP) "$$(cat $(WINDOWS_OUT)/.e2e-path)" $(WINHOST):"$(WINWORK)/e2e.exe"
	$(SSH) $(WINHOST) "cd $(subst /,\\,$(WINWORK)) && set PATH=$(subst /,\\,$(WINWORK));%PATH% && e2e.exe --ignored --test-threads=1"

## Fetch the fixture generator onto the Windows machine, once.
##
## Beside the tests, never into the system: ffmpeg makes fixtures and
## judges output, and neither is something the shipped program does.
windows-ffmpeg:
	$(SSH) $(WINHOST) "if not exist \"$(subst /,\\,$(WINWORK))\" mkdir \"$(subst /,\\,$(WINWORK))\""
	$(SSH) $(WINHOST) "cd $(subst /,\\,$(WINWORK)) && curl -L -s -o ff.zip $(FFMPEG_WIN_URL) && tar -xf ff.zip && for /d %d in (ffmpeg-*) do move %d\\bin\\ffmpeg.exe . & move %d\\bin\\ffprobe.exe ."
	@printf 'windows ffmpeg: ' && $(SSH) $(WINHOST) "$(subst /,\\,$(WINWORK))\\ffmpeg.exe -version" | head -1

## Long-form gates: the ones that need real bulk rather than a GPU.
## Two of them, both measured against our own demuxer:
##   * byte scale — writes ~9 GiB transiently (a 4.6 GiB spool, then the
##     output) to prove co64 chunk offsets and a 64-bit mdat;
##   * table scale — a 12 h, 7-track movie (13 M samples) to prove the
##     sample tables stay run-length collapsed and the mux fits in RAM.
## Release build on purpose: these are scale tests, and debug turns 13 M
## samples from seconds into minutes without testing anything more.
test-long:
	$(CARGO) test --release -p sovmr-mp4 --test mux -- --ignored --nocapture

## Install the binary, icon, and pin-able launcher entry for the current
## user; registers the right-click "Compress with sovmr" file-manager
## action via the desktop entry's MimeType list.
install: build
	install -Dm755 $(BIN) $(BINDIR)/sovmr
	install -Dm644 dist/icons/sovmr.svg $(ICONDIR)/sovmr.svg
	install -d $(DESKTOPDIR)
	sed "s|@BINDIR@|$(BINDIR)|" dist/sovmr.desktop \
		> $(DESKTOPDIR)/sovmr.desktop
	-update-desktop-database $(DESKTOPDIR) 2>/dev/null
	@echo "installed: $(BINDIR)/sovmr, icon, and $(DESKTOPDIR)/sovmr.desktop"
	@echo "sovmr is now pinnable in the launcher and offered for video files"
	@printf 'linux:   ' && $(BINDIR)/sovmr --version

uninstall:
	rm -f $(BINDIR)/sovmr $(DESKTOPDIR)/sovmr.desktop $(ICONDIR)/sovmr.svg

## Install the freshly cross-built program on the Windows machine and
## report what landed there. A fix is not delivered until both machines
## have it, and this is the half that used to be done by hand.
##
## The installer carries the program inside itself, so exactly one file
## needs to travel.
deploy-windows: windows
	$(SSH) $(WINHOST) "if not exist \"$(subst /,\\,$(WINWORK))\" mkdir \"$(subst /,\\,$(WINWORK))\""
	$(SCP) $(WINDOWS_OUT)/sovmr-setup.exe $(WINHOST):"$(WINWORK)/sovmr-setup.exe"
	$(SSH) $(WINHOST) "\"$(subst /,\\,$(WINWORK))\\sovmr-setup.exe\""
	@printf 'windows: ' && ssh $(WINHOST) "%LOCALAPPDATA%\\Programs\\sovmr\\sovmr.exe --version"

## Which build is on which machine — the question a two-machine project
## has to answer without guessing from file dates.
##
## Every probe is bounded. A build older than the `--version` flag treats
## it as a filename and opens the GUI, which would hang this target
## forever; a machine that is switched off would do the same. Both are
## ordinary states, so both are reported rather than waited on.
PROBE := timeout 15
versions:
	@printf 'linux:   ' && ($(PROBE) $(BINDIR)/sovmr --version 2>/dev/null \
		|| echo "not installed, or too old to say (pre 2.0.0-rc.1)")
	@printf 'windows: ' && ($(PROBE) $(SSH) $(WINHOST) \
		"%LOCALAPPDATA%\\Programs\\sovmr\\sovmr.exe --version" 2>/dev/null \
		|| echo "unreachable, not installed, or too old to say")
	@printf 'source:  ' && echo "sovmr $$(sed -n 's/^version = .\(.*\)./\1/p' Cargo.toml | head -1) ($$(git rev-parse --short HEAD)$$(git diff --quiet || printf -- -dirty))"

$(APPIMAGETOOL):
	mkdir -p dist/tools
	curl -L --fail -o $(APPIMAGETOOL) $(APPIMAGETOOL_URL)
	chmod +x $(APPIMAGETOOL)

## Sovereign portable package: one stripped fat-LTO binary in a standard
## .AppDir, squashed into a single-file AppImage. No bundled FFmpeg, no
## bundled anything: the engine IS the binary (dlopen'ing the NVIDIA
## driver's own libraries at runtime). Needs nothing from the host but a
## display and the NVIDIA driver.
package: build $(APPIMAGETOOL)
	rm -rf $(APPDIR)
	install -Dm755 $(BIN) $(APPDIR)/usr/bin/sovmr
	strip $(APPDIR)/usr/bin/sovmr
	install -Dm644 dist/icons/sovmr.svg $(APPDIR)/sovmr.svg
	install -Dm644 dist/icons/sovmr.svg \
		$(APPDIR)/usr/share/icons/hicolor/scalable/apps/sovmr.svg
	sed "s|@BINDIR@/||" dist/sovmr.desktop > $(APPDIR)/sovmr.desktop
	ln -sf sovmr.svg $(APPDIR)/.DirIcon
	printf '#!/bin/sh\n# APPDIR is exported by the AppImage runtime; the fallback covers\n# direct AppDir launches. No external commands on the runtime path,\n# so this works even with a crippled PATH.\nexec "$${APPDIR:-$$(dirname "$$0")}/usr/bin/sovmr" "$$@"\n' \
		> $(APPDIR)/AppRun
	chmod 755 $(APPDIR)/AppRun
	ARCH=x86_64 $(APPIMAGETOOL) --appimage-extract-and-run $(APPDIR) $(APPIMAGE)
	@echo "single-file package ready:"
	@ls -lh $(APPIMAGE) | awk '{print "  " $$9 " (" $$5 ")"}'

clean:
	$(CARGO) clean
	rm -rf $(APPDIR) $(APPIMAGE)
