# Face Recognition Attendance — check in/out with YuNet + SFace

A few known faces are enrolled in the first seconds of the run and given
neutral labels (Person 1, Person 2, ...). Afterwards every detected face
is matched against the enrolled gallery by cosine similarity: a match
above the threshold checks that person **IN**, absence for 2 s checks
them **OUT**, and a face with no match is flagged **UNKNOWN FACE**. A
live attendance board (name, IN/OUT, first seen, total time present,
check-ins) is drawn under the video, and a timestamped attendance log +
end-of-run summary are written out.

## Consent — read this first

**Enrol only people who have agreed to be recognised.** Face
recognition is a biometric: in a real deployment, enrol staff or
students who have consented, tell them what is stored and why, and keep
their embeddings private. The demo clip uses free-license stock footage;
the people in it are strangers who did not consent to enrolment, so they
are labelled only as Person 1 / Person 2 for demonstration, their
embeddings exist only for the length of the run, and nothing is stored.

## What it does

- **Detection:** YuNet (OpenCV DNN, `face_detection_yunet_2023mar.onnx`,
  233 KB, shipped in this folder) — the same detector as the
  face-blur-privacy-shield project.
- **Recognition:** SFace embeddings (OpenCV DNN,
  `face_recognition_sface_2021dec.onnx`, 38.7 MB — **not** shipped, see
  Weights below), L2-normalised, so the dot product is the cosine
  similarity. Match threshold **0.50**, chosen from measurement, not
  tuned to flatter the demo: on the demo clip, same-person cosines run
  0.64–0.98 and cross-person cosines 0.06–0.29, so 0.50 sits in a wide
  empty gap.
- **Enrolment:** the first 2 s of the run. A face seen in at least 3
  detection rounds enrols with the mean of its embeddings. Faces that
  appear later are never enrolled mid-run — they read UNKNOWN.
- **Attendance:** CHECK-IN the moment a person is recognised, CHECK-OUT
  after 2 s unseen, session-end check-out for anyone still IN. Present
  time is time recognised in frame.
- **Speed:** detection + embedding runs every 5th frame, boxes held
  between rounds, so it stays usable on a shared CPU.

## Demo (real footage)

`face-recognition-attendance-demo.mp4` — free-license Pexels footage,
**"People Talking Inside the Conference Room"** (Pexels video 5941111),
three colleagues talking around a table. The clip is 15.6 s, looped to
30 s (loop points at 13.0 s and 26.0 s; tracks reset there, attendance
totals kept — logged). Verified 16 Nov 2026 — see
`face-recognition-attendance-demo_summary.txt` for the exact counters:

- 900 frames, a face in 900/900 (100%), peak 3 faces in one frame
- 2,010 face-frames: 1,680 known, 210 unknown
- Person 1 and Person 2 enrolled at 2.0 s, both CHECK-IN at 2.0 s,
  present 28.0 s each, CHECK-OUT at session end
- UNKNOWN FACE at 9.5 s and again at 22.5 s (the second loop) — the
  woman on the left, seen only in profile/partially, best match 0.30,
  below the 0.50 threshold
- Accepted-match cosine: min 0.64 / median 0.89 / max 0.98 (n=336);
  best cross-person cosine: median 0.10 / max 0.29; unknown-face best
  cosine: 0.20–0.30

H.264 copy in `../project-demos/linkedin-h264/` (codec-verified h264,
yuv420p, faststart); frames inspected at 5 s, 10 s and 20 s — green
boxes with name + score for recognised faces, a red UNKNOWN FACE box
for the unrecognised one, board underneath.

## Simulation (internal logic check only — NOT the demo)

`--simulate` renders a meeting room with three seated figures; the
third walks in at 10 s. The simulation's face source finds them with a
skin-colour mask + contours, and the embedding, matching, check-in/out
and logging code is the exact same code as real mode. Verified
16 Nov 2026 (30 s run): 900 frames, 2,400 face-frames, both enrolled
figures CHECK-IN at 2.0 s and stay IN 28.0 s, 2 check-ins, 2 check-outs,
0 unknown. **Known sim limit:** rendered faces are flat drawings, so
their embeddings sit close together (same-person cosine 0.95–1.00,
cross-person 0.87–0.89) and the late figure matches Person 1 instead of
reading UNKNOWN. The simulation therefore verifies enrolment,
check-in/out, the board, the log and the summary — the UNKNOWN flag and
identity discrimination are verified on real footage instead, where the
scores separate cleanly (see above).

## Bugs found and fixed during verification

1. **SFace was fed a bare bounding box** (`[x1, y1, x2, y2]`) instead of
   YuNet's full 15-value row with landmarks. `alignCrop` then produced
   a constant crop: every embedding came out identical (all pairwise
   cosines 1.00), Person 2 never checked in, and no UNKNOWN ever fired.
   Fixed by passing the landmarks through — the score stats in the
   summary are what caught it.
2. **An orange shirt was detected as a face** by the simulation's
   skin-colour mask (its hue sits inside the skin HSV range). The shirt
   was recoloured outside that range.
3. **Enrolment-phase faces were counted as "unknown"** in the counters
   (nobody is known yet in the first 2 s). They are now excluded until
   enrolment closes.
4. **"First seen" was recorded at enrolment-close time** (2.0 s) instead
   of when the face actually appeared (0.0 s). Now taken from the track.

## Honest limitations

- A face that is turned away, occluded or very small may be missed, and
  a profile face may score below the threshold and read UNKNOWN — the
  demo shows both (the left woman has no box at all while her back is
  to the camera, then a red UNKNOWN box when her profile appears).
- "Distinct UNKNOWN faces" counts track segments, not people: the same
  unrecognised woman counts twice in the demo because the clip loops.
- Present time is time recognised on camera, not proof of attendance —
  someone out of frame is not counted, and a look-alike could in
  principle cross the threshold (the measured cross-person max is 0.29,
  well below 0.50, but no threshold is a guarantee).
- Recognition quality depends on pose and lighting; frontal, well-lit
  faces match at 0.9+, a three-quarter profile at 0.65–0.75.

## Weights

- `face_detection_yunet_2023mar.onnx` (233 KB, from the OpenCV model
  zoo) ships in this folder.
- `face_recognition_sface_2021dec.onnx` (38.7 MB) is **not** shipped —
  GitHub's web upload caps a file at 25 MB. The script downloads it once
  on first run from the OpenCV model zoo and caches it next to the
  script (it is in `.gitignore` so it never gets committed).

## Usage

```bash
pip install -r requirements.txt
python face_recognition_attendance.py                    # file picker
python face_recognition_attendance.py --source meet.mp4
python face_recognition_attendance.py --source 0         # webcam
python face_recognition_attendance.py --simulate --max-seconds 30 --no-display
python face_recognition_attendance.py --source clip.mp4 --loop --max-seconds 30
```

Press Q to quit. Outputs next to `--out` (default `output.mp4`):
`output.mp4`, `output_summary.txt`, `output_attendance.log`.
`--threshold` changes the match cosine (default 0.50);
`--enrol-seconds` changes the enrolment window (default 2.0).

## Files

- `face_recognition_attendance.py` — detector + recogniser + attendance
  board + renderer + simulation
- `face_detection_yunet_2023mar.onnx` — YuNet face detector (233 KB)
- `requirements.txt` — opencv-python-headless, numpy
- `README.md` — this file
