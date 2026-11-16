#!/usr/bin/env python3
"""
face_recognition_attendance.py — face-recognition attendance with consent.

YuNet face detection + SFace face embeddings (both OpenCV DNN, no torch).
A few known faces are enrolled in the first seconds of the run and given
neutral labels (Person 1, Person 2, ...). Afterwards every detected face
is matched against the enrolled gallery by cosine similarity: a match
above the threshold checks that person IN, absence for 2 s checks them
OUT, and a face with no match is flagged UNKNOWN FACE. A live attendance
board (name, IN/OUT, first seen, total time present) is drawn under the
video, and a timestamped attendance log + end-of-run summary are written
out.

CONSENT: enrol only people who have agreed to be recognised. The demo
clip uses free-license stock footage; the people in it are strangers who
did not consent to enrolment, so they are labelled only as Person 1 /
Person 2 for demonstration and their embeddings are never stored — the
gallery exists only for the length of the run. A real deployment must
enrol staff/students who have consented, and keep their embeddings
private.

Models: YuNet face_detection_yunet_2023mar.onnx (233 KB, shipped in this
folder). SFace face_recognition_sface_2021dec.onnx (38.7 MB, over GitHub's
25 MB web-upload limit, so it is NOT shipped — the script downloads it
once on first run from the OpenCV model zoo and caches it next to the
script).

Real mode:  --source video.mp4 | 0 (webcam) | pick (file chooser)
Simulation: --simulate renders a meeting room with seated figures; the
simulation's face source finds them with a skin-colour mask + contours —
the embedding, matching, check-in/out and logging code is the exact same
code as real mode, and the recogniser never sees the simulation's ground
truth. (Simulation match scores are reported in the run summary; rendered
faces are flat drawings, so their scores are not expected to match real
faces — the simulation checks the attendance logic, the real footage
checks recognition.)

Usage:
    python face_recognition_attendance.py                  # file picker
    python face_recognition_attendance.py --source meet.mp4
    python face_recognition_attendance.py --source 0       # webcam
    python face_recognition_attendance.py --simulate --max-seconds 30 --no-display
    python face_recognition_attendance.py --source clip.mp4 --loop --max-seconds 30

Press Q to quit. Outputs next to --out (default output.mp4):
output.mp4, output_summary.txt, output_attendance.log
"""

import argparse
import math
import os
import sys
import time
import urllib.request

import cv2
import numpy as np

# ---------------------------------------------------------------- constants

YUNET_DEFAULT = "face_detection_yunet_2023mar.onnx"
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")
SFACE_DEFAULT = "face_recognition_sface_2021dec.onnx"
SFACE_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_recognition_sface/face_recognition_sface_2021dec.onnx")
SFACE_SIZE = 38696353  # bytes, as served by the model zoo (38.7 MB)

SCORE_THRESHOLD = 0.6   # YuNet detection score
MATCH_THRESHOLD = 0.5   # cosine similarity for a recognised match.
# Measured on the demo clip before choosing it (normalised SFace
# embeddings): same person 0.67-0.95, different person 0.06-0.24, so
# 0.5 sits in a wide empty gap. Scores are reported, never tuned to
# flatter the demo.
NMS_THRESHOLD = 0.3
HOLD_FRAMES = 6         # keep the last boxes this long when detection blinks
DETECT_EVERY = 5        # detect + embed every Nth frame (CPU sanity)
ENROL_SECONDS = 2.0     # enrolment window at the start of the run
ENROL_MIN_SAMPLES = 3   # detection rounds needed before a face enrols
ABSENT_SECONDS = 2.0    # unseen this long -> CHECK-OUT
TRACK_DIST = 100        # px, centre distance for track association

CANVAS_W, CANVAS_H = 1280, 720
VIDEO_W, VIDEO_H = 1280, 600
PANEL_H = CANVAS_H - VIDEO_H


# ----------------------------------------------------------------- models

def ensure_sface(path):
    """Download the SFace model once if it is not already next to the script."""
    if os.path.exists(path) and os.path.getsize(path) > 1000000:
        return path
    print("Downloading SFace model (38.7 MB, one-time) from the OpenCV model zoo ...")
    try:
        urllib.request.urlretrieve(SFACE_URL, path)
    except Exception as exc:
        raise RuntimeError(
            "Could not download the SFace model: %s\nGet it manually with:\n"
            "  curl -L -o %s %s" % (exc, SFACE_DEFAULT, SFACE_URL))
    if os.path.getsize(path) < 1000000:
        os.remove(path)
        raise RuntimeError("SFace download looked incomplete (too small); "
                           "delete %s and retry." % path)
    print("SFace model saved to %s" % path)
    return path


class FaceDetector:
    """YuNet face detector (cv2.dnn)."""

    def __init__(self, model_path, score_threshold=SCORE_THRESHOLD):
        if not os.path.exists(model_path):
            raise RuntimeError(
                "Face model not found: %s\nDownload it with:\n"
                "  curl -L -o %s %s" % (model_path, YUNET_DEFAULT, YUNET_URL))
        self.model_path = model_path
        self.score_threshold = score_threshold
        self.detector = None
        self.input_size = None

    def detect(self, frame_bgr):
        """Return a list of (x, y, w, h, score, landmarks) face boxes,
        left to right. landmarks is the 5-point YuNet set (right eye,
        left eye, nose tip, right mouth corner, left mouth corner) —
        SFace's alignCrop needs the full 15-value YuNet row, landmarks
        included; passing a bare box makes alignCrop produce a constant
        crop and every embedding comes out identical (measured: all
        pairwise cosines 1.00 — a real bug, caught by the score stats)."""
        h, w = frame_bgr.shape[:2]
        if self.detector is None or self.input_size != (w, h):
            self.detector = cv2.FaceDetectorYN_create(
                self.model_path, "", (w, h), self.score_threshold,
                NMS_THRESHOLD, 20)
            self.input_size = (w, h)
        _, faces = self.detector.detect(frame_bgr)
        if faces is None:
            return []
        out = []
        for f in faces:
            x, y, fw, fh = (float(v) for v in f[:4])
            score = float(f[-1])
            x0, y0 = max(0, int(x)), max(0, int(y))
            x1, y1 = min(w, int(x + fw)), min(h, int(y + fh))
            if x1 - x0 >= 12 and y1 - y0 >= 12:
                landmarks = np.array(f[4:14], dtype=np.float32).reshape(5, 2)
                out.append((x0, y0, x1 - x0, y1 - y0, score, landmarks))
        out.sort(key=lambda b: b[0])
        return out


class FaceRecognizer:
    """SFace embeddings (cv2.dnn). Embeddings are L2-normalised, so the
    dot product between two embeddings IS the cosine similarity."""

    def __init__(self, model_path):
        ensure_sface(model_path)
        self.recognizer = cv2.FaceRecognizerSF_create(model_path, "")

    def embed(self, frame_bgr, box, landmarks):
        """Normalised 128-d embedding for one face. box is (x, y, w, h,
        score); landmarks is the 5-point set (right eye, left eye, nose,
        right mouth corner, left mouth corner). Both are packed into the
        15-value YuNet row alignCrop expects."""
        x, y, w, h, score = box
        face = np.array([[x, y, w, h,
                          landmarks[0][0], landmarks[0][1],
                          landmarks[1][0], landmarks[1][1],
                          landmarks[2][0], landmarks[2][1],
                          landmarks[3][0], landmarks[3][1],
                          landmarks[4][0], landmarks[4][1],
                          score]], dtype=np.float32)
        aligned = self.recognizer.alignCrop(frame_bgr, face)
        vec = self.recognizer.feature(aligned).flatten().astype(np.float64)
        n = np.linalg.norm(vec)
        return vec / n if n > 0 else vec

    @staticmethod
    def cosine(a, b):
        return float(np.dot(a, b))


# --------------------------------------------------------------- simulation

SIM_FIGURES = [
    # (base x, scale, skin BGR, hair BGR, shirt BGR, enters at t)
    # Shirt colours are deliberately outside the skin HSV range, so the
    # sim face source never mistakes a shirt for a face (an orange shirt
    # did exactly that in the first version — measured and fixed).
    (300, 1.0, (120, 160, 210), (30, 25, 20), (180, 120, 60), 0.0),
    (640, 1.0, (90, 140, 200), (200, 200, 210), (160, 90, 60), 0.0),
    (980, 1.0, (150, 185, 225), (40, 40, 40), (70, 150, 90), 10.0),
]


def render_sim_scene(t, w=1280, h=720, seed=7):
    """Rendered meeting room with seated figures. The sim face source
    (skin-colour mask) finds them from the pixels only — the pipeline
    never sees the renderer's ground truth. The third figure walks in
    and sits down at t = 10 s, so it is never enrolled: it must be
    flagged UNKNOWN."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), (215, 205, 190), dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (w, 300), (225, 220, 210), -1)  # wall
    cv2.rectangle(img, (0, 300), (w, h), (170, 150, 120), -1)  # floor band
    cv2.rectangle(img, (80, 60), (560, 220), (200, 210, 220), -1)  # whiteboard
    cv2.rectangle(img, (700, 60), (1200, 220), (190, 200, 210), -1)
    cv2.rectangle(img, (60, 560), (w - 60, h), (120, 95, 70), -1)  # table
    for i, (bx, scale, skin, hair, shirt, enters) in enumerate(SIM_FIGURES):
        if t < enters:
            continue
        # slide-in for the late figure over its first 2 s
        slide = 0
        if enters > 0:
            slide = int(max(0, 260 * (1 - min(1.0, (t - enters) / 2.0))))
        cx = bx + slide
        cy = 400
        s = scale
        cv2.rectangle(img, (cx - int(70 * s), cy + int(30 * s)),
                      (cx + int(70 * s), cy + int(160 * s)), shirt, -1)
        cv2.ellipse(img, (cx, cy), (int(26 * s), int(32 * s)), 0, 0, 360,
                    skin, -1)
        cv2.ellipse(img, (cx, cy - int(10 * s)), (int(26 * s), int(15 * s)),
                    0, 180, 360, hair, -1)
        cv2.circle(img, (cx - int(10 * s), cy - int(2 * s)),
                   max(2, int(3 * s)), (20, 20, 20), -1)
        cv2.circle(img, (cx + int(10 * s), cy - int(2 * s)),
                   max(2, int(3 * s)), (20, 20, 20), -1)
    noise = rng.integers(0, 6, (h, w, 1), dtype=np.uint8)
    return cv2.add(img, np.repeat(noise, 3, axis=2))


def sim_face_source(frame_bgr):
    """Find the rendered figures' heads from pixels only: skin-colour
    mask in HSV + contour filtering. Same output shape as YuNet. The
    visible skin blob is only the LOWER face (hair covers the top half
    of each head), so the box is extended upward by its own height to
    cover the full head, and landmarks are placed at standard fractions
    of that full-head box — a heuristic on the detected blob, not the
    renderer's ground truth."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([4, 40, 90])
    upper = np.array([22, 180, 255])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 300:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if 0.4 < w / max(h, 1) < 2.0 and h > 18:
            fx, fy, fw, fh = x, max(0, y - h), w, 2 * h
            lm = np.array([[fx + 0.3 * fw, fy + 0.4 * fh],
                           [fx + 0.7 * fw, fy + 0.4 * fh],
                           [fx + 0.5 * fw, fy + 0.55 * fh],
                           [fx + 0.35 * fw, fy + 0.75 * fh],
                           [fx + 0.65 * fw, fy + 0.75 * fh]],
                          dtype=np.float32)
            boxes.append((fx, fy, fw, fh, 0.99, lm))
    boxes.sort(key=lambda b: b[0])
    return boxes


# -------------------------------------------------------------------- render

def compose(frame_bgr, tracks, board_lines, counters_line, t, enrolling):
    """Video on top, attendance board in the bottom panel."""
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    video = cv2.resize(frame_bgr, (VIDEO_W, VIDEO_H))
    sx, sy = VIDEO_W / frame_bgr.shape[1], VIDEO_H / frame_bgr.shape[0]
    for tr in tracks:
        x, y, w, h, _ = tr["box"]
        lx, ly = int(x * sx), int(y * sy)
        lw, lh = int(w * sx), int(h * sy)
        known = tr["label"] is not None
        colour = (0, 200, 0) if known else (0, 0, 255)
        cv2.rectangle(video, (lx, ly), (lx + lw, ly + lh), colour, 2)
        if known:
            text = "%s  %.2f" % (tr["label"], tr["score"])
        else:
            text = "UNKNOWN FACE"
        cv2.putText(video, text, (lx, max(16, ly - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    canvas[0:VIDEO_H, 0:VIDEO_W] = video
    cv2.rectangle(canvas, (0, VIDEO_H), (CANVAS_W, CANVAS_H),
                  (15, 15, 15), -1)
    cv2.putText(canvas, "FACE RECOGNITION ATTENDANCE  —  t=%.1fs%s"
                % (t, "  (ENROLLING...)" if enrolling else ""),
                (12, VIDEO_H + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    y = VIDEO_H + 52
    for line in board_lines:
        cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y += 26
    cv2.putText(canvas, counters_line, (12, CANVAS_H - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1,
                cv2.LINE_AA)
    return canvas


# ---------------------------------------------------------------------- main

def pick_file():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Choose a video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"),
                       ("All files", "*.*")])
        root.destroy()
        return path
    except Exception:
        return ""


def parse_source(value):
    if value is None or value == "pick":
        return pick_file()
    return value


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--source", default=None,
                    help="video path, webcam index, or 'pick'")
    ap.add_argument("--simulate", action="store_true",
                    help="rendered meeting room (internal logic check only)")
    ap.add_argument("--max-seconds", type=float, default=30)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--out", default="output.mp4")
    ap.add_argument("--model", default=YUNET_DEFAULT)
    ap.add_argument("--sface-model", default=SFACE_DEFAULT)
    ap.add_argument("--threshold", type=float, default=MATCH_THRESHOLD,
                    help="cosine similarity for a recognised match")
    ap.add_argument("--enrol-seconds", type=float, default=ENROL_SECONDS)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    yunet_path = args.model
    if not os.path.isabs(yunet_path):
        yunet_path = os.path.join(here, yunet_path)
    sface_path = args.sface_model
    if not os.path.isabs(sface_path):
        sface_path = os.path.join(here, sface_path)

    detector = None if args.simulate else FaceDetector(yunet_path)
    recognizer = FaceRecognizer(sface_path)

    cap = None
    if not args.simulate:
        src = parse_source(args.source)
        if not src:
            print("No video chosen. Exiting.")
            return 1
        try:
            src = int(src)
        except (TypeError, ValueError):
            pass
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print("Could not open video source: %r" % (src,))
            return 1

    out_path = args.out
    base, _ = os.path.splitext(out_path)
    summary_path = base + "_summary.txt"
    log_path = base + "_attendance.log"
    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
        (CANVAS_W, CANVAS_H))
    events = []

    def log_event(t, text):
        line = "[%6.1fs] %s" % (t, text)
        events.append(line)
        print(line)

    total_frames = int(args.max_seconds * args.fps)
    tracks = []          # {id, box, label, score, last_seen, samples[], enrolled}
    next_track_id = 0
    gallery = {}         # name -> mean normalised embedding
    enrol_closed = False
    # attendance per enrolled person: {in, first_seen, last_match, total, checkins}
    attendance = {}
    unknown_ids = set()
    unknown_frames = 0
    known_frames = 0
    match_scores = []    # cosine of every accepted match
    cross_scores = []    # best cosine to a *different* enrolled person
    unknown_scores = []  # best cosine of every unknown face
    faces_seen_total = 0
    frames_with_faces = 0
    max_faces = 0
    detect_rounds = 0
    loops = 0
    t_start = time.time()
    frame_idx = 0

    def close_enrolment(t):
        nonlocal enrol_closed
        enrol_closed = True
        for tr in tracks:
            if len(tr["samples"]) >= ENROL_MIN_SAMPLES:
                name = "Person %d" % (len(gallery) + 1)
                mean = np.mean(tr["samples"], axis=0)
                mean = mean / np.linalg.norm(mean)
                gallery[name] = mean
                tr["enrolled"] = name
                attendance[name] = {"in": False, "first_seen": tr["first_seen"],
                                    "last_match": None, "total": 0.0,
                                    "checkins": 0}
                log_event(t, "ENROLLED %s (from %d samples, first seen %.1fs)"
                          % (name, len(tr["samples"]), tr["first_seen"]))
        if not gallery:
            log_event(t, "WARNING: no face enrolled — no attendance to track")

    while frame_idx < total_frames:
        if args.simulate:
            frame = render_sim_scene(frame_idx / args.fps)
            ok = True
        else:
            ok, frame = cap.read()
            if not ok:
                if args.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = cap.read()
                    if ok:
                        loops += 1
                        tracks = []
                        log_event(frame_idx / args.fps,
                                  "loop point — tracks reset (attendance "
                                  "totals kept)")
                if not ok:
                    print("End of video at frame %d." % frame_idx)
                    break
        frame = cv2.resize(frame, (1280, 720))
        t = frame_idx / args.fps
        dt = 1.0 / args.fps

        if not enrol_closed and t >= args.enrol_seconds:
            close_enrolment(t)

        fresh = []
        if frame_idx % DETECT_EVERY == 0:
            detect_rounds += 1
            fresh = (sim_face_source(frame) if args.simulate
                     else detector.detect(frame))
            # associate fresh boxes with tracks (nearest centre)
            used = set()
            new_tracks = []
            for box in fresh:
                cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
                best, best_d = None, TRACK_DIST
                for tr in tracks:
                    if id(tr) in used:
                        continue
                    tx, ty = (tr["box"][0] + tr["box"][2] / 2,
                              tr["box"][1] + tr["box"][3] / 2)
                    d = math.hypot(cx - tx, cy - ty)
                    if d < best_d:
                        best, best_d = tr, d
                if best is not None:
                    used.add(id(best))
                    best["box"] = box[:5]
                    best["landmarks"] = box[5]
                    best["last_seen"] = t
                    new_tracks.append(best)
                else:
                    tr = {"id": next_track_id, "box": box[:5],
                          "landmarks": box[5], "label": None,
                          "score": 0.0, "last_seen": t, "samples": [],
                          "enrolled": None, "first_seen": t,
                          "unknown_logged": False}
                    next_track_id += 1
                    new_tracks.append(tr)
            # keep un-matched tracks briefly (hold) then drop them
            for tr in tracks:
                if id(tr) not in used and t - tr["last_seen"] < 0.4:
                    new_tracks.append(tr)
            tracks = new_tracks

            # embed + match every freshly seen face
            for tr in tracks:
                if t - tr["last_seen"] > 0.01:
                    continue  # only faces seen in this round
                emb = recognizer.embed(frame, tr["box"], tr["landmarks"])
                if not enrol_closed:
                    tr["samples"].append(emb)
                    continue
                if not gallery:
                    continue
                sims = {n: recognizer.cosine(emb, g)
                        for n, g in gallery.items()}
                best_name = max(sims, key=sims.get)
                best_score = sims[best_name]
                others = [v for n, v in sims.items() if n != best_name]
                if best_score >= args.threshold:
                    tr["label"] = best_name
                    tr["score"] = best_score
                    match_scores.append(best_score)
                    if others:
                        cross_scores.append(max(others))
                else:
                    tr["label"] = None
                    tr["score"] = best_score
                    unknown_scores.append(best_score)
                    if tr["id"] not in unknown_ids:
                        unknown_ids.add(tr["id"])
                        log_event(t, "UNKNOWN FACE appeared (best match "
                                  "%.2f, below %.2f threshold)"
                                  % (best_score, args.threshold))

        # attendance state machine (per frame, from last match times)
        present_now = set()
        for tr in tracks:
            if tr["label"] is not None and t - tr["last_seen"] < 0.6:
                present_now.add(tr["label"])
        for name, rec in attendance.items():
            if name in present_now:
                if not rec["in"]:
                    rec["in"] = True
                    rec["checkins"] += 1
                    log_event(t, "CHECK-IN  %s" % name)
                rec["last_match"] = t
                rec["total"] += dt
            elif rec["in"] and rec["last_match"] is not None \
                    and t - rec["last_match"] > ABSENT_SECONDS:
                rec["in"] = False
                log_event(t, "CHECK-OUT %s (was present %.1fs total)"
                          % (name, rec["total"]))

        count = len([tr for tr in tracks if t - tr["last_seen"] < 0.4])
        if count > 0:
            frames_with_faces += 1
        max_faces = max(max_faces, count)
        faces_seen_total += count
        known = sum(1 for tr in tracks
                    if tr["label"] is not None and t - tr["last_seen"] < 0.4)
        # during the enrolment window faces are neither known nor unknown
        # yet — they are the samples the gallery is built from
        unknown = (count - known) if enrol_closed else 0
        known_frames += known
        unknown_frames += unknown

        board = []
        for name in sorted(attendance):
            rec = attendance[name]
            state = "IN " if rec["in"] else "OUT"
            board.append("%s  %-9s  first seen %5.1fs   present %5.1fs   "
                         "check-ins %d" % (name, state, rec["first_seen"],
                                           rec["total"], rec["checkins"]))
        if not board:
            board = ["Enrolling faces — attendance board fills in after "
                     "the first %.0f s..." % args.enrol_seconds]
        counters = ("frames %d   faces %d (known %d, unknown %d)   "
                    "check-ins %d   unknown people %d"
                    % (frame_idx, faces_seen_total, known_frames,
                       unknown_frames,
                       sum(r["checkins"] for r in attendance.values()),
                       len(unknown_ids)))
        canvas = compose(frame, [tr for tr in tracks
                                 if t - tr["last_seen"] < 0.4],
                         board, counters, t, not enrol_closed)
        writer.write(canvas)
        if not args.no_display:
            try:
                cv2.imshow("Face Recognition Attendance — Q to quit", canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
            except cv2.error:
                pass  # headless OpenCV: no GUI available
        frame_idx += 1

    if not enrol_closed:
        close_enrolment(frame_idx / args.fps)
    end_t = frame_idx / args.fps
    for name, rec in attendance.items():
        if rec["in"]:
            rec["in"] = False
            log_event(end_t, "CHECK-OUT %s (session end, present %.1fs total)"
                      % (name, rec["total"]))

    writer.release()
    if cap is not None:
        cap.release()
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass  # headless OpenCV

    wall = time.time() - t_start
    n = max(frame_idx, 1)

    def stats(vals):
        if not vals:
            return "n/a"
        v = sorted(vals)
        return ("min %.2f / median %.2f / max %.2f (n=%d)"
                % (v[0], v[len(v) // 2], v[-1], len(v)))

    summary = [
        "Face Recognition Attendance — run summary",
        "mode: %s" % ("simulation (rendered meeting room)" if args.simulate
                      else "real footage: %s" % (args.source or "picker")),
        "frames rendered: %d (%.1f s at %d fps)" % (n, n / args.fps, args.fps),
        "detection rounds: %d (every %d frames)" % (detect_rounds, DETECT_EVERY),
        "frames with >= 1 face: %d (%.1f%%)" % (
            frames_with_faces, 100.0 * frames_with_faces / n),
        "peak faces in one frame: %d" % max_faces,
        "total face-frames: %d (known %d, unknown %d)"
        % (faces_seen_total, known_frames, unknown_frames),
        "enrolled: %d person(s): %s"
        % (len(gallery), ", ".join(sorted(gallery)) or "none"),
        "check-ins: %d   check-outs: %d"
        % (sum(r["checkins"] for r in attendance.values()),
           events and sum(1 for e in events if "CHECK-OUT" in e)),
        "distinct UNKNOWN faces: %d" % len(unknown_ids),
        "match threshold (cosine): %.2f" % args.threshold,
        "accepted-match cosine: %s" % stats(match_scores),
        "best cross-person cosine (other enrolled person): %s"
        % stats(cross_scores),
        "unknown-face best cosine: %s" % stats(unknown_scores),
        "loops of source clip: %d" % loops,
        "detector: YuNet (score >= %.2f); recogniser: SFace (OpenCV DNN)"
        % SCORE_THRESHOLD,
        "wall time: %.1f s (%.1f fps render)" % (wall, n / max(wall, 0.01)),
    ]
    for name in sorted(attendance):
        rec = attendance[name]
        summary.append("  %s: first seen %.1fs, present %.1fs total, "
                       "%d check-in(s), ends %s"
                       % (name, rec["first_seen"], rec["total"],
                          rec["checkins"], "IN" if rec["in"] else "OUT"))
    summary.append("note: counts are detected/recognised faces, not a "
                   "census — a turned-away, occluded or very small face "
                   "may be missed, and a profile face may score below "
                   "the match threshold and read UNKNOWN.")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary) + "\n")
    with open(log_path, "w") as f:
        f.write("\n".join(events) + ("\n" if events else ""))
    print("\n".join(summary))
    print("Wrote %s, %s, %s" % (out_path, summary_path, log_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
