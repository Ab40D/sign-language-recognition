# ✋ ASL-Based Hand-Sign Recognition Prototype for Device Control

**ASL ON/OFF sign → Computer Vision → Command → USB Serial → Arduino → LED**

> [!IMPORTANT]
> **This project recognizes two predefined ASL signs and maps them to device
> commands. It does not understand ASL grammar, sentences, facial
> expressions, or general sign language.** It is a small accessibility /
> human-computer-interaction prototype, not an ASL translator.

This repository is the evolution of a voice-controlled Arduino LED project
(Version 1, preserved in [`legacy_voice/`](legacy_voice/README.md)). The
microphone is gone; the input is now a **webcam watching your hand**. The
serial protocol, the Arduino communication code, and the Arduino-side
command handling were reused and extended — only the human input changed.

```text
ASL HAND SIGN (ON / OFF)
        ↓
     WEBCAM
        ↓
     OPENCV (frame capture)
        ↓
     MEDIAPIPE (21 hand landmarks)
        ↓
     ML CLASSIFIER (Random Forest on normalized landmarks)
        ↓
     ON / OFF  (+ confidence)
        ↓
     COMMAND MAPPER
        ↓
     LIGHT_ON / LIGHT_OFF
        ↓
     USB SERIAL (9600 baud)
        ↓
     ARDUINO
        ↓
     GREEN LED (on) / RED LED (off)
```

---

## ✨ Project Overview

Hold the ASL sign for **ON** in front of your webcam → the green LED turns
on. Hold the sign for **OFF** → the red LED turns on. That is the whole
product.

The pipeline is deliberately split into small, replaceable layers:

| Layer | Module | Responsibility |
| ----- | ------ | -------------- |
| Hand detection | `vision/hand_detector.py` | Run MediaPipe, return 21 landmarks per hand |
| Normalization | `vision/preprocessing.py` | Make landmarks position/scale/rotation/hand independent |
| Feature extraction | `vision/feature_extractor.py` | One hand detection → one 63-number vector |
| Data collection | `recognition/collect_data.py` | Save labeled landmark samples from your webcam |
| Training | `recognition/train_classifier.py` | Train + evaluate + save the Random Forest |
| Classifier | `recognition/classifier.py` | Load the model, predict ON/OFF + confidence |
| Command mapping | `control/command_parser.py` | `ON → LIGHT_ON`, `OFF → LIGHT_OFF` (nothing else) |
| Stability | `control/debounce.py` | Send each command **once** per stable sign |
| Serial link | `control/serial_controller.py` | Connect, send, receive, reconnect (or simulate) |
| Firmware | `arduino/sign_controller/sign_controller.ino` | Parse one line, drive two LEDs, answer one line |
| Application | `main.py` | Wire everything together + on-screen UI |

---

## ♿ Accessibility

This prototype explores **one more way** to control a device without voice,
touch, or a keyboard. Voice assistants are not usable or comfortable for
everyone — for example people who cannot speak, people in quiet or noisy
environments, or people who simply prefer a visual-gestural interface.

Two honest notes:

* **Not all Deaf people use ASL** (or any sign language), and not everyone
  who signs can perform every handshape. A two-sign prototype helps nobody
  by pretending otherwise.
* Sign languages are **real, complete languages** and are **not universal**:
  ASL (used here) differs from LSF, BSL, and every other sign language.

---

## 🤟 ASL Reference (which signs and why)

Primary reference: **HandSpeak** — <https://www.handspeak.com/>

* ON: <https://www.handspeak.com/word/1553/>
* OFF: <https://www.handspeak.com/word/1536/>
* LIGHT (context, e.g. `LIGHT #ON` / `LIGHT #OFF`): <https://www.handspeak.com/word/1272/>

HandSpeak documents **two variants** for each word:

1. **The two-flat-hands variant.**
   *ON*: dominant palm-down flat hand rests on top of the non-dominant
   palm-down flat hand. *OFF*: the dominant hand then moves outward and
   downward off the non-dominant hand. HandSpeak itself notes the isolated
   word form is rarely used in real ASL, and — critically for this project —
   ON and OFF share the same static pose and differ **only in movement
   direction**. A single-frame landmark classifier cannot see movement.
2. **The lexicalized fingerspelled loans `#ON` and `#OFF`.**
   HandSpeak states that `#ON` is *"commonly used by Deaf native-signers
   more than the other sign"*, and that `#OFF` is commonly used in
   constructions like `TURN+#OFF` and `LIGHT+#OFF` — exactly the
   "turn the light on/off" context this prototype targets.

**Selected signs (this prototype):**

| Command | Sign | What you hold for the camera |
| ------- | ---- | ---------------------------- |
| ON | `#ON` | Fingerspell O→N quickly and **hold the final N handshape**: index + middle fingers extended, resting over the tucked thumb; ring + pinky curled into the palm. |
| OFF | `#OFF` | Fingerspell O→F→F quickly and **hold the final F handshape**: thumb tip touching index tip forming a circle; middle, ring and pinky fingers extended and spread. |

Why these two: they are the documented, native-preferred forms for the
light-switch context, they are **one-handed**, and their held positions are
**statically distinct** (an N-like hold vs. an F hold), which is what a
per-frame landmark classifier can actually separate. The movement-based
flat-hand variants are documented above for completeness.

---

## 🏗️ Architecture

```text
┌────────────┐   RGB frame   ┌──────────────┐  21×(x,y,z)  ┌───────────────┐
│   WEBCAM   │ ────────────► │   MediaPipe  │ ───────────► │ preprocessing │
└────────────┘               │ HandLandmarker│              │ (normalize)   │
                             └──────────────┘              └──────┬────────┘
                                                                  │ 63 floats
                ┌───────────────────────────────────────────────┘
                ▼
     ┌────────────────────┐   ON/OFF + confidence   ┌─────────────────┐
     │  Random Forest     │ ───────────────────────►│ SignStabilizer  │
     │  (scikit-learn)    │                         │ (debounce)      │
     └────────────────────┘                         └───────┬─────────┘
                                                            │ stable sign
                ┌───────────────────────────────────────────┘
                ▼
     ┌────────────────────┐  LIGHT_ON / LIGHT_OFF   ┌─────────────────┐
     │  command_parser    │ ───────────────────────►│ SerialController│
     │  (sign → command)  │                         │ (pyserial 9600) │
     └────────────────────┘                         └───────┬─────────┘
                                                            │ USB
                             ┌──────────────────────────────┘
                             ▼
                  ┌─────────────────────┐      ┌───────────────────┐
                  │  Arduino sketch     │ ───► │ D8 GREEN / D9 RED │
                  │  sign_controller    │ ◄─── │      LEDs         │
                  └─────────────────────┘      └───────────────────┘
                     replies: READY, LIGHT_STATE:ON/OFF,
                              ERROR:UNKNOWN_COMMAND
```

### How each stage works (the short course)

1. **MediaPipe hand detection** — a two-stage model: a light palm detector
   finds the hand region, then a regression network outputs 21 3D landmark
   points plus handedness. We never feed pixels to our own classifier.
2. **Landmarks → features** (`vision/preprocessing.py`) — raw landmarks
   depend on hand position, distance and tilt. Normalization removes that:
   mirror left hands to a canonical side → move wrist to origin → divide by
   hand size → rotate so the middle finger points "up" → flatten to 63
   numbers. **The exact same function runs at collection, training and
   inference** — there is only one copy of it.
3. **Classifier** — a Random Forest over those 63 numbers. Chosen because it
   is strong on small tabular datasets, needs no feature scaling, returns
   class probabilities (our confidence score), and predicts in microseconds.
4. **Stability** (`control/debounce.py`) — a command fires only when the
   same sign was predicted in `stable_frames` consecutive frames **and**
   held for `min_hold_seconds`, **and** differs from the last sign sent,
   **and** the `cooldown_seconds` window has passed. Holding ON for five
   seconds sends `LIGHT_ON` exactly once.
5. **Serial protocol** — one ASCII line per message, 9600 8N1, identical to
   the voice version: `LIGHT_ON`, `LIGHT_OFF`, `STATUS` in; `READY`,
   `LIGHT_STATE:ON`, `LIGHT_STATE:OFF`, `ERROR:UNKNOWN_COMMAND` out.
6. **Arduino** — the sketch never trusts a bad line: unknown input changes
   nothing and returns `ERROR:UNKNOWN_COMMAND`.

---

## 📦 Installation

Requirements: **Python 3.10+**, an Arduino Uno (or clone), a webcam.

```bash
git clone https://github.com/Ab40D/voice-controlled-arduino.git
cd voice-controlled-arduino

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# MediaPipe model weights (~8 MB, one time):
python scripts/fetch_hand_landmarker.py
```

`requirements.txt`: `opencv-python`, `mediapipe`, `numpy`, `scikit-learn`,
`joblib`, `pyserial` — nothing else.

> Note: current mediapipe (≥ 0.10.35 / 1.x) uses the **Tasks API** used
> here; the old `mp.solutions` API was removed upstream.

Upload the firmware with the Arduino IDE:
open `arduino/sign_controller/sign_controller.ino` and upload it to the board.

---

## 📚 Dataset

There is **no bundled image dataset**, and that is deliberate:

* No suitably-licensed public dataset provides exactly these two held
  signs (`#ON`, `#OFF`) as isolated poses.
* Landmark-based training needs only ~60 samples per class, and samples of
  **your** hand, camera and lighting are worth more than any scraped set.
* Random Google Images would be an unverified training source — rejected by
  design.

All training data is collected by you, in about one minute (next section).
References (HandSpeak pages above) are used only to learn/verify the signs.

## 🎥 Data Collection

Learn the two holds (see ASL Reference), then from the repository root:

```bash
python recognition/collect_data.py --label ON      # ~60 samples, 30 seconds
python recognition/collect_data.py --label OFF
```

A window opens with a 3-second countdown, then captures ~10 samples/second
while showing your landmarks. Keep **exactly one hand** visible; vary
position and distance a little between captures. Press `Q` to abort.
Sessions are stored as `data/samples/<LABEL>/session_<timestamp>.npz`
(normalized 63-dim landmark vectors + labels). Repeat any time to add more.

## 🧠 Training

```bash
python recognition/train_classifier.py
```

Loads every session of both labels, fails clearly if data is missing or too
small (< 10 per class), does a stratified 80/20 split, trains a 200-tree
Random Forest, prints accuracy / per-class precision-recall / confusion
matrix, and saves `recognition/model/sign_classifier.pkl`.

## ▶️ Running

```bash
python main.py                     # auto-detects the Arduino port
python main.py --port /dev/ttyACM0   # Linux explicit port
python main.py --port COM3           # Windows explicit port
python main.py --simulate          # no Arduino needed (simulated replies)
python main.py --list-ports        # see available serial ports
python main.py --send LIGHT_ON     # one-shot serial test (no model needed)
```

Useful flags: `--camera N`, `--confidence 0.7`, `--stable-frames 6`,
`--cooldown 1.0`.

Keyboard (development only): `O` inject ON · `F` inject OFF · `C` clear
state · `Q` quit.

The window shows the webcam image with landmarks and a status panel:
detected sign, confidence, last command, Arduino status
(`CONNECTED` / `SIMULATION` / `DISCONNECTED`) and the last Arduino response.

---

## 🔌 Arduino Wiring

**Safety: this project only switches low-voltage LEDs with current-limiting
resistors. Never connect Arduino GPIO to mains voltage.**

```text
D8  ──── 220 Ω ────► GREEN LED anode (long leg)     cathode ──── GND
D9  ──── 220 Ω ────► RED   LED anode (long leg)     cathode ──── GND
```

| State | Green (D8) | Red (D9) |
| ----- | ---------- | -------- |
| `LIGHT_ON`  | ON  | OFF |
| `LIGHT_OFF` | OFF | ON  |

---

## 🧪 Tests

Headless unit + integration tests (no webcam/Arduino needed; a synthetic
landmark generator exercises the full train→predict→stabilize→serial chain):

```bash
python -m unittest discover -s tests
```

---

## 🛠️ Troubleshooting

| Problem | Fix |
| ------- | --- |
| `MediaPipe model file not found` | Run `python scripts/fetch_hand_landmarker.py` |
| `Webcam unavailable` | Plug in the camera, close other apps using it, try `--camera 1` |
| `No Arduino serial port found` | Plug in the board, use a **data** USB cable, run `--list-ports`, or pass `--port` |
| `Permission denied` on Linux | `sudo usermod -aG dialout $USER`, then log out/in (or `sudo chmod a+rw /dev/ttyACM0` for now) |
| Port busy | Close the Arduino IDE Serial Monitor (`fuser -v /dev/ttyACM0`) |
| No `READY` line | Correct sketch uploaded? Baud is 9600. Cable data-capable? |
| Sign never triggers | Better lighting, one hand only, hold the pose steady ~1 s; lower `--confidence` only if you understand the trade-off |
| Wrong sign recognized | Collect more samples of both signs (varied positions) and retrain |
| Commands repeat | Should not happen — check you didn't set `--stable-frames 1 --cooldown 0` |

---

## ⚠️ Limitations

* **Two signs only.** No vocabulary, no grammar, no fingerspelling, no
  facial expressions, no continuous signing. This is a command trigger, not
  a translator.
* **Static holds.** Signs distinguished mainly by movement are out of scope
  for single-frame landmark classification.
* **Personal model.** The classifier is trained on one person's hand,
  camera and lighting. A different user should collect their own dataset.
* **ASL only.** Nothing here generalizes to other sign languages.
* **Single hand.** Two visible hands are treated as "no command".

## 🚀 Future Improvements (not implemented)

* More ASL command signs (e.g. UP/DOWN for dimming)
* Continuous sign recognition with temporal models (movement-aware)
* Text-display commands / multi-device control
* MQTT broker instead of direct USB
* Home Assistant integration
* KNX / smart-building integration

---

## 📁 Project Structure

```text
├── main.py                     # real-time application (UI + wiring)
├── config.py                   # all tunable settings in one place
├── vision/
│   ├── hand_detector.py        # MediaPipe Tasks wrapper
│   ├── preprocessing.py        # THE shared landmark normalization
│   ├── feature_extractor.py    # detection -> 63-dim feature vector
│   └── draw.py                 # landmark skeleton drawing
├── recognition/
│   ├── collect_data.py         # webcam dataset collector
│   ├── train_classifier.py     # train / evaluate / save
│   ├── classifier.py           # model loading + prediction
│   └── model/sign_classifier.pkl   (created by training)
├── control/
│   ├── command_parser.py       # ON->LIGHT_ON, OFF->LIGHT_OFF
│   ├── serial_controller.py    # ArduinoLink + auto-detect + simulator
│   └── debounce.py             # SignStabilizer state machine
├── arduino/sign_controller/sign_controller.ino
├── models/hand_landmarker.task # MediaPipe weights (fetch script)
├── data/samples/               # your collected dataset (gitignored)
├── scripts/fetch_hand_landmarker.py
├── tests/                      # unit + integration tests
├── legacy_voice/               # preserved Version 1 (voice project)
├── requirements.txt
└── README.md
```

---

*This project recognizes two predefined ASL signs and maps them to device
commands. It does not understand ASL grammar, sentences, facial expressions,
or general sign language.*
