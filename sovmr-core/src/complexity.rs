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

//! Reading a film's shape out of its sample sizes.
//!
//! # The free measurement
//!
//! An encoder spends bits where a picture is hard. So the size of each
//! coded frame *is* a record of how hard the source encoder found it —
//! a complexity curve already computed, already stored, and free to
//! read: `stsz` is a table of integers at the front of the file.
//!
//! Every other way of learning the same thing costs a decode. This
//! costs a table walk, which is why it is worth having even though it
//! is an approximation.
//!
//! # What it is not
//!
//! It is the *previous* encoder's opinion, shaped by its own settings.
//! A frame is large because that encoder chose to spend there, which is
//! evidence about the picture and not proof. So this is used to place
//! decisions — where a scene begins, which stretch deserves more — and
//! never to claim a quality.
//!
//! # Why the median and not the mean
//!
//! The thing being looked for is the outlier. A mean moves towards
//! whatever it is asked to detect; a median does not, so a single
//! enormous frame cannot raise the bar that would have caught it.

/// How far above its neighbourhood a frame must sit to be called a cut.
///
/// Two and a half times the local median. A new scene has nothing to
/// predict from, so its first frame is coded almost from scratch and
/// stands far above the inter-coded frames around it. Ordinary motion
/// moves the number by tens of percent; a cut moves it by multiples.
const CUT_RATIO_MILLI: u64 = 2500;

/// Frames either side that form the neighbourhood a frame is judged
/// against.
///
/// About a second at ordinary rates: long enough that one hard frame
/// does not define its own baseline, short enough that a genuinely busy
/// passage raises the bar it is judged by rather than being reported as
/// cut after cut.
const WINDOW: usize = 24;

/// The least a stretch can be before it is worth calling a scene.
///
/// Under this, a flicker of hard frames would be split into scenes a
/// second long, and a keyframe on each would cost more than the
/// placement saves.
const MIN_SCENE: usize = 12;

/// A film's shape, read from its coded sizes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Complexity {
    /// Coded bytes per frame, in decode order.
    sizes: Vec<u32>,
}

impl Complexity {
    /// Read a curve from a track's sample sizes.
    pub fn from_sizes(sizes: Vec<u32>) -> Complexity {
        Complexity { sizes }
    }

    /// Frames measured.
    pub fn len(&self) -> usize {
        self.sizes.len()
    }

    /// Whether there is nothing to judge.
    pub fn is_empty(&self) -> bool {
        self.sizes.is_empty()
    }

    /// Frame indices where a new scene appears to begin.
    ///
    /// Always includes frame zero: the film begins somewhere, and a
    /// caller placing keyframes needs the first one.
    ///
    /// A frame is a cut when it stands [`CUT_RATIO_MILLI`] above the
    /// median of its neighbourhood, and when the previous cut is at
    /// least [`MIN_SCENE`] frames behind — a picture that changes twice
    /// in half a second changed once as far as a keyframe is concerned.
    pub fn scene_cuts(&self) -> Vec<usize> {
        if self.sizes.is_empty() {
            return Vec::new();
        }
        let mut cuts = vec![0usize];
        for index in 1..self.sizes.len() {
            let Some(median) = self.local_median(index) else {
                continue;
            };
            if median == 0 {
                continue;
            }
            let ratio = u64::from(self.sizes[index]) * 1000 / u64::from(median);
            let far_enough = index - cuts[cuts.len() - 1] >= MIN_SCENE;
            if ratio >= CUT_RATIO_MILLI && far_enough {
                cuts.push(index);
            }
        }
        cuts
    }

    /// Median coded size of the frames around `index`, excluding it.
    ///
    /// Excluded on purpose: a frame must not contribute to the baseline
    /// it is being judged against, or a large enough one justifies
    /// itself.
    fn local_median(&self, index: usize) -> Option<u32> {
        let from = index.saturating_sub(WINDOW);
        let to = (index + WINDOW + 1).min(self.sizes.len());
        let mut around: Vec<u32> = self.sizes[from..to]
            .iter()
            .enumerate()
            .filter(|(offset, _)| from + offset != index)
            .map(|(_, size)| *size)
            .collect();
        if around.is_empty() {
            return None;
        }
        around.sort_unstable();
        Some(around[around.len() / 2])
    }

    /// How hard each scene is relative to the film as a whole, in
    /// thousandths.
    ///
    /// A thousand is exactly average. A caller distributing bits can
    /// weigh a stretch by this without knowing anything about what is in
    /// it — which is the point: no decode, no pixels, no guesswork about
    /// content.
    pub fn scene_weights(&self) -> Vec<(usize, usize, u64)> {
        let cuts = self.scene_cuts();
        if cuts.is_empty() {
            return Vec::new();
        }
        let total: u64 = self.sizes.iter().map(|s| u64::from(*s)).sum();
        if total == 0 || self.sizes.is_empty() {
            return Vec::new();
        }
        let average = total / self.sizes.len() as u64;
        if average == 0 {
            return Vec::new();
        }
        let mut scenes = Vec::with_capacity(cuts.len());
        for (n, start) in cuts.iter().enumerate() {
            let end = cuts.get(n + 1).copied().unwrap_or(self.sizes.len());
            let frames = end - start;
            if frames == 0 {
                continue;
            }
            let sum: u64 = self.sizes[*start..end].iter().map(|s| u64::from(*s)).sum();
            scenes.push((*start, end, (sum / frames as u64) * 1000 / average));
        }
        scenes
    }

    /// How unevenly the film's difficulty is spread, in thousandths.
    ///
    /// A thousand means every scene is equally hard. Above that, a few
    /// scenes carry most of the cost while the rest are cheap.
    ///
    /// This is what a single average bitrate cannot express and why a
    /// ceiling set from one is either too mean for the hard scenes or too
    /// generous for the easy ones. A concentrated film can be given a lower
    /// average and still look the same, because the bits it needs are
    /// needed in fewer places.
    ///
    /// Measured as the ratio of the mean scene weight to the median. Mean
    /// over median, not variance: variance answers in squared bytes and has
    /// to be scaled by something before it means anything, while this ratio
    /// is already the number a ceiling wants to be divided by.
    pub fn concentration_milli(&self) -> u64 {
        let scenes = self.scene_weights();
        if scenes.len() < 2 {
            // One scene is perfectly even by definition, and so is nothing.
            return 1000;
        }
        let mut weights: Vec<u64> = scenes.iter().map(|(_, _, w)| *w).collect();
        weights.sort_unstable();
        let median = weights[weights.len() / 2];
        if median == 0 {
            // Half the film costs nothing measurable: as concentrated as it
            // gets, and saying so beats dividing by zero.
            return u64::from(u32::MAX);
        }
        let mean: u64 = weights.iter().sum::<u64>() / weights.len() as u64;
        mean * 1000 / median
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A calm stretch: every frame about the same size.
    fn calm(frames: usize, size: u32) -> Vec<u32> {
        (0..frames)
            .map(|i| size + (i % 5) as u32 * (size / 50).max(1))
            .collect()
    }

    #[test]
    fn a_scene_change_stands_out_of_its_neighbourhood() {
        // What the whole thing is for. Two calm stretches joined by one
        // frame that had nothing to predict from.
        let mut sizes = calm(60, 2000);
        sizes.push(30_000);
        sizes.extend(calm(60, 2000));
        let cuts = Complexity::from_sizes(sizes).scene_cuts();
        assert!(cuts.contains(&0), "the film starts somewhere: {cuts:?}");
        assert!(
            cuts.contains(&60),
            "the cut at frame 60 was missed: {cuts:?}"
        );
        assert_eq!(cuts.len(), 2, "and nothing else was called a cut: {cuts:?}");
    }

    #[test]
    fn ordinary_motion_is_not_a_scene_change() {
        // Frames that vary by tens of percent are a moving camera, not a
        // cut. A detector that called those cuts would ask for a
        // keyframe every few frames and make the file larger.
        let sizes: Vec<u32> = (0..200).map(|i| 2000 + ((i * 37) % 900) as u32).collect();
        let cuts = Complexity::from_sizes(sizes).scene_cuts();
        assert_eq!(cuts, vec![0], "motion was mistaken for cuts: {cuts:?}");
    }

    #[test]
    fn a_frame_cannot_justify_itself() {
        // The median excludes the frame being judged. Including it would
        // let a large enough frame raise its own baseline — and the
        // larger the frame, the better it would hide.
        let mut sizes = calm(60, 1000);
        sizes.push(1_000_000);
        sizes.extend(calm(60, 1000));
        let cuts = Complexity::from_sizes(sizes).scene_cuts();
        assert!(cuts.contains(&60), "an enormous frame hid itself: {cuts:?}");
    }

    #[test]
    fn two_changes_within_half_a_second_are_one_scene() {
        // A keyframe costs more than the placement saves when scenes are
        // a few frames long.
        let mut sizes = calm(60, 2000);
        sizes.push(30_000);
        sizes.extend(calm(4, 2000));
        sizes.push(30_000);
        sizes.extend(calm(60, 2000));
        let cuts = Complexity::from_sizes(sizes).scene_cuts();
        assert_eq!(
            cuts.len(),
            2,
            "a flicker was split into separate scenes: {cuts:?}"
        );
    }

    #[test]
    fn a_busy_passage_raises_the_bar_it_is_judged_by() {
        // Hard material is not a sequence of cuts. The neighbourhood
        // median rises with it, which is exactly why it is local.
        let mut sizes = calm(60, 2000);
        sizes.extend(calm(120, 25_000));
        let cuts = Complexity::from_sizes(sizes).scene_cuts();
        assert!(
            cuts.len() <= 2,
            "a busy passage was read as {} cuts: {cuts:?}",
            cuts.len()
        );
    }

    #[test]
    fn scene_weights_say_which_stretch_is_hard() {
        // The number a rate planner needs: how much harder than average
        // this stretch is, with no decode and no pixels.
        let mut sizes = calm(60, 1000);
        sizes.push(30_000);
        sizes.extend(calm(60, 8000));
        let scenes = Complexity::from_sizes(sizes).scene_weights();
        assert_eq!(scenes.len(), 2, "{scenes:?}");
        assert!(
            scenes[1].2 > scenes[0].2 * 3,
            "the hard stretch scored {} against the easy one's {}",
            scenes[1].2,
            scenes[0].2
        );
    }

    #[test]
    fn an_even_film_and_a_lopsided_one_are_told_apart() {
        // The number a single average bitrate cannot express. Both films
        // below cost the same in total; one spends it evenly and one
        // spends it in a corner, and a ceiling set from the average is
        // wrong for at least one of them.
        let mut even = Vec::new();
        for _ in 0..6 {
            even.push(40_000);
            even.extend(calm(60, 4000));
        }
        let mut lopsided = Vec::new();
        lopsided.push(40_000);
        lopsided.extend(calm(60, 20_000));
        for _ in 0..5 {
            lopsided.push(40_000);
            lopsided.extend(calm(60, 800));
        }
        let even = Complexity::from_sizes(even).concentration_milli();
        let lopsided = Complexity::from_sizes(lopsided).concentration_milli();
        assert!(
            even < 1400,
            "an evenly spread film scored {even}, which is not even"
        );
        assert!(
            lopsided > even * 2,
            "lopsided scored {lopsided} against even's {even} — the measure \
             cannot tell them apart"
        );
    }

    #[test]
    fn a_single_scene_is_even_by_definition() {
        assert_eq!(
            Complexity::from_sizes(calm(200, 3000)).concentration_milli(),
            1000
        );
        assert_eq!(
            Complexity::from_sizes(Vec::new()).concentration_milli(),
            1000
        );
    }

    #[test]
    fn a_film_whose_median_scene_costs_nothing_is_answered_not_divided_by() {
        // Half the scenes measurably free is as concentrated as it gets,
        // and it is also a division by zero if nobody looked.
        let mut sizes = Vec::new();
        sizes.push(60_000);
        sizes.extend(calm(60, 9000));
        for _ in 0..4 {
            sizes.push(60_000);
            sizes.extend(vec![0u32; 60]);
        }
        // Answered, not divided by. The fixture does not in fact reach a
        // zero median — a run of empty frames gives its neighbours a
        // median of zero, so no cut is called there and the scenes never
        // separate — which is worth knowing and is why the assertion is
        // about the property rather than a number I expected.
        let c = Complexity::from_sizes(sizes).concentration_milli();
        assert!(c > 0, "a concentration of zero is not an answer");
    }

    #[test]
    fn nothing_to_judge_is_answered_rather_than_divided_by() {
        assert!(Complexity::from_sizes(Vec::new()).scene_cuts().is_empty());
        assert!(Complexity::from_sizes(Vec::new())
            .scene_weights()
            .is_empty());
        // A track of empty frames divides by zero everywhere if the
        // guards are missing.
        let empty = Complexity::from_sizes(vec![0; 100]);
        assert_eq!(empty.scene_cuts(), vec![0]);
        assert!(empty.scene_weights().is_empty());
    }

    #[test]
    fn one_frame_is_one_scene() {
        let single = Complexity::from_sizes(vec![5000]);
        assert_eq!(single.scene_cuts(), vec![0]);
        assert_eq!(single.scene_weights().len(), 1);
        assert_eq!(single.len(), 1);
        assert!(!single.is_empty());
    }
}
