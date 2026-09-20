# Medical evidence-span RULEBOOK (annotation convention of question_train.csv)

Derived 19 Sep 2026 from all 39 training conversations (195 gold-yes questions) against the production
transcripts (faster-whisper large-v3-turbo word times) and speech islands (gap 0.15 s, thr -35 dBFS).
Convention dump with every gold span next to its islands: scratchpad med/convention.txt (see med_pack.md).

## 1. What the evidence is
- The conversations are stitched TTS utterance clips. One gold span = one clip (island) or a run of
  consecutive clips: 115/195 golds are exactly ONE island, 52 are 2 islands, 28 are 3-6 islands (gap 0.15).
  Island-range oracle (best consecutive island run, fitted edges) = mean tIoU 0.891.
- The evidence is the ONE passage that states the fact asked about most explicitly, usually the DOCTOR's
  explicit statement/finding/plan ("Your chest and heart both sound normal", "The plan is to continue it
  as needed", "I am creating prescriptions for both Pamol and Ibumedin now"). The PATIENT's line is the
  evidence when the question is about what the patient reports, asks or wants ("Did the patient ask for
  penicillin?" -> "I want penicillin for it"; "Does the patient want the skin changes removed?" -> the
  patient's "I would still rather have them off").
- FIRST explicit mention wins: only 11/195 golds have an earlier island that also matches the question,
  22/195 have a later restatement that is NOT chosen. The exceptions are when the earlier mention is
  vague or partial and the later one is the explicit finding (sample_17 "Will the treatment last two
  weeks?" -> the patient's full recap "After a meal, every day, for 2 weeks", not the doctor's clipped
  "100 mg daily for / 2 weeks"). Rule of thumb: pick the island whose words answer the question on their
  own; prefer the earliest such island; do not pick a vague foreshadowing ("Let us go through your
  results") over the concrete statement.
- Several questions can share one gold span (sample_19: three questions about the renewed medicines all
  map to "Activel, Aromir, and Isomeprazole. All three renewed.").

## 2. Multi-island spans (80/195)
- 64/80 are a CONTINUATION: the fact runs across clips of the same speaker ("For the sinuses, /
  penicillin. / One million international units, / four times daily, / for seven days."). Include every
  clip that carries part of the asked fact (dose + frequency + duration when the question is about the
  prescription; only the "for seven days" tail when the question is about the duration ... the range is
  chosen so that it covers the asked fact and little else: sample_6 "Has penicillin been prescribed for
  seven days?" = islands 35..38 (drops "For the sinuses,"), "Are four daily doses ... planned?" = 34..38).
- 16/80 are a QUESTION + ANSWER exchange: when the answer clip is elliptical ("Yes.", "None.", "It
  does.") the gold starts at the preceding question clip ("And you are still taking pantoprazole
  alongside it? / Yes, every day with it."; "So the medicine is genuinely working? / Yes."). When the
  answer is self-contained ("No, nothing new.", "No redness at all.") the question clip is NOT included
  (sample_10 islands 15..16 only). A doctor's confirmation after a patient statement ("Exactly that.")
  is not included either.
- Never include the acknowledgement that follows ("Good.", "All right.", "Understood.").

## 3. Sub-island spans (26/195, mean loss 0.06 tIoU when you take the whole island)
- When one island holds two coordinated facts ("Your blood pressure is normal, and your foot status is
  normal."; "Overall, your diabetes is stable with no signs of complications."; "The plan is advice on
  diet and on exercise.") the gold is ONLY the clause that answers the question: end the span before
  ", and ..." for the first clause, start at the clause's first word for the second clause. Word-level
  trims (w0/w1 in the choice) raise the oracle from 0.891 to 0.949, so use them whenever the island says
  more than the asked fact.
- Leading hedges/fillers are kept when they belong to the clause ("my assessment is that your diabetes
  is stable" keeps "assessment is that"); a leading "Overall," / "From the picture," / "and" is dropped.
- A span never starts inside a word and never ends before the clause's last content word.

## 4. Boundaries (seconds)
- START = the acoustic onset of the first word, NOT whisper's word start: whisper first-word starts are
  early by 0.28 s median (deciles 0.04..0.66). gold_start - energy_onset(first word) = -0.03 median
  (deciles -0.20..+0.11). Fitted: start = onset - 0.04 when an energy onset (spans.energy_onsets, -50 dBFS)
  falls inside the first word (175/195), else whisper first-word start + 0.00.
- END = whisper last-word end - 0.02 (median gold_end - word_end = -0.02, deciles -0.26..+0.05); the
  clip's decay before the next pause is NOT included.
- Fitted rule on the oracle island range (gap 0.15): mean tIoU 0.891 (raw word edges 0.844; gap 0.3: 0.850).
  medical/offline/rule.json holds it; spans_from_choice.py applies it.
- Gold spans are 0.16-14.2 s, median 2.9 s. Two golds are annotation defaults at [0.00-0.26]/[0.00-0.16]
  (the fact is not in the audio); nothing predicts those.

## 5. Question types
- positive (195, answer yes): as above. Negation/normality wording ("Was erythema migrans absent?",
  "Are the findings unremarkable?", "Will the treatment be continued unchanged?") is answered by the
  explicit negative/normal statement ("There is no erythema migrans", "The examination I have done is
  normal" ... through "Nothing abnormal was found"), tIoU 0.856 vs 0.902 for the rest: these spans more
  often run over a Q+A pair ("Any redness around the site? / No redness at all. / Any swelling? / No
  swelling either." is one gold of 3 islands for "Are redness and swelling absent?").
- hard_negative (142, answer no, no span): a plausible fact that contradicts or was never stated: a
  wrong number/drug/site ("Was the prescribed dose 200 mg daily?", "Was Pantoprazole the acid-reducing
  drug that was renewed?"), the opposite finding ("Were abnormal sounds heard over the lungs?", "Has the
  bite area become red and swollen?"), an action that was declined/not planned ("Will the patient be
  referred to a pain clinic?", "Does the patient want to avoid antibiotics?"). They sit right next to a
  positive twin: check the exact value/entity before answering yes.
- off_topic (53, answer no): everyday small talk that is NOT in the conversation (pets, cooking,
  vehicles, trips, hobbies, cameras). Note the small talk that IS there (weather, waiting-room
  magazines) is never asked as a yes question in training.
- Answer balance is exactly 50/50 over the set; per conversation 3-7 yes.
- The portal scores the span on every gold-yes row even if we answer no: ALWAYS give a span (the
  best-matching island) for no answers too.

## 6. Procedure for an annotator (per conversation pack)
1. Read the islands once; note who speaks (doctor/patient alternate per clip in most conversations).
2. For each question: find the earliest island whose words state the asked fact explicitly; extend
   over following clips only while they still carry part of the asked fact; prepend the question clip
   only when the answer clip is a bare yes/no; trim to the clause with w0/w1 when the island holds more.
3. hard_negative twins: verify the exact number/drug/side; off_topic: answer no; still give a span for
   every question (the closest island) because no-spans are scored on gold-yes rows.
4. Write choices as {"<sample_N>": {"<q index>": 7 | [7, 9] | {"i0":7,"i1":9,"w0":2,"w1":-2} | null}}.
