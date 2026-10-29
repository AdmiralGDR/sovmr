// SPDX-License-Identifier: GPL-3.0-or-later
//
// sovmr — sovereign media re-encoder
// Copyright (C) 2026  AdmiralGDR
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

//! The generic seam between a renderer and an engine.
//!
//! [`UiBridge`] owns exactly two things: the *receiving* end of an
//! [`EngineEvent`] channel and an opaque command callback. It does not know
//! what produces the events (a local scheduler thread today, a socket to a
//! remote encode farm tomorrow) nor what consumes the commands — which is
//! precisely what makes `sovmr-ui` reusable for a totally different app.
//!
//! [`EngineEvent`]: sovmr_core::queue::messages::EngineEvent

use std::sync::mpsc::Receiver;

use sovmr_core::queue::messages::{EngineCommand, EngineEvent};

use crate::state::DashboardState;

/// Channel plumbing handed to the shell (or any other renderer).
///
/// # Example
///
/// ```
/// use std::sync::mpsc::channel;
/// use sovmr_core::job::JobId;
/// use sovmr_core::queue::messages::{EngineCommand, EngineEvent};
/// use sovmr_ui::bridge::UiBridge;
/// use sovmr_ui::state::DashboardState;
///
/// let (event_tx, event_rx) = channel();
/// let (cmd_tx, cmd_rx) = channel();
/// let bridge = UiBridge::new(event_rx, move |cmd| { let _ = cmd_tx.send(cmd); });
///
/// // Engine side (normally sovmr_core::queue::scheduler) emits events...
/// let id = JobId::next();
/// event_tx.send(EngineEvent::Queued { id, input_name: "a.mp4".into(), insight: None }).unwrap();
///
/// // ...the renderer folds them once per frame, without blocking...
/// let mut dashboard = DashboardState::default();
/// assert_eq!(bridge.pump(&mut dashboard), 1);
/// assert_eq!(dashboard.rows.len(), 1);
///
/// // ...and user actions travel the other way.
/// bridge.dispatch(EngineCommand::Cancel(id));
/// assert!(matches!(cmd_rx.try_recv(), Ok(EngineCommand::Cancel(_))));
/// ```
pub struct UiBridge {
    events: Receiver<EngineEvent>,
    commands: Box<dyn Fn(EngineCommand) + Send>,
}

impl UiBridge {
    /// Wire a bridge from the engine's event stream and a command sink.
    ///
    /// `commands` is any closure; for the local scheduler pass
    /// `move |cmd| engine.send(cmd)`. It must never block, because it is
    /// invoked from inside the render loop.
    pub fn new(
        events: Receiver<EngineEvent>,
        commands: impl Fn(EngineCommand) + Send + 'static,
    ) -> Self {
        UiBridge {
            events,
            commands: Box::new(commands),
        }
    }

    /// Drain every event that has arrived since the last frame into
    /// `state`. Non-blocking by construction (`try_recv`); returns how many
    /// events were folded, which doubles as a "something changed" signal
    /// for repaint scheduling.
    pub fn pump(&self, state: &mut DashboardState) -> usize {
        let mut n = 0;
        while let Ok(event) = self.events.try_recv() {
            state.apply(event);
            n += 1;
        }
        n
    }

    /// Send one command to the engine. Fire-and-forget; effects come back
    /// as events on the next [`UiBridge::pump`].
    pub fn dispatch(&self, cmd: EngineCommand) {
        (self.commands)(cmd);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sovmr_core::job::JobId;
    use std::sync::mpsc::channel;

    #[test]
    fn pump_is_nonblocking_on_empty_channel() {
        let (_event_tx, event_rx) = channel();
        let bridge = UiBridge::new(event_rx, |_| {});
        let mut state = DashboardState::default();
        assert_eq!(bridge.pump(&mut state), 0); // returns immediately
    }

    #[test]
    fn pump_folds_all_pending_events_in_order() {
        let (event_tx, event_rx) = channel();
        let bridge = UiBridge::new(event_rx, |_| {});
        let id = JobId::next();
        event_tx
            .send(EngineEvent::Queued {
                id,
                input_name: "x.mp4".into(),
                insight: None,
            })
            .unwrap();
        event_tx.send(EngineEvent::Started { id }).unwrap();
        event_tx.send(EngineEvent::Cancelled { id }).unwrap();
        let mut state = DashboardState::default();
        assert_eq!(bridge.pump(&mut state), 3);
        assert!(state.rows[0].status.is_terminal());
    }

    #[test]
    fn pump_survives_disconnected_engine() {
        let (event_tx, event_rx) = channel();
        let bridge = UiBridge::new(event_rx, |_| {});
        drop(event_tx); // engine died
        let mut state = DashboardState::default();
        assert_eq!(bridge.pump(&mut state), 0); // no panic, no hang
    }

    #[test]
    fn dispatch_reaches_the_sink() {
        let (_event_tx, event_rx) = channel();
        let (cmd_tx, cmd_rx) = channel();
        let bridge = UiBridge::new(event_rx, move |c| {
            let _ = cmd_tx.send(c);
        });
        bridge.dispatch(EngineCommand::CancelAll);
        assert!(matches!(cmd_rx.try_recv(), Ok(EngineCommand::CancelAll)));
    }
}
