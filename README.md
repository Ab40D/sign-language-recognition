# ASL-Based Hand-Sign Recognition for Device Control

**A computer vision and machine learning prototype that translates two predefined ASL handshape inputs into Arduino-controlled device commands.**

**Hand Sign → Computer Vision → Machine Learning → Serial Communication → Arduino**

## Overview

This project explores an alternative approach to human–computer interaction by using static handshape recognition to control a physical device.

A webcam captures the user's hand, MediaPipe extracts hand landmarks, and a machine learning classifier predicts one of two predefined classes: `ON` or `OFF`. After the prediction passes stability checks, the application sends a corresponding command to an Arduino over USB serial communication.

The prototype demonstrates how computer vision, lightweight machine learning, and embedded systems can work together in an accessible, real-time control pipeline.

> **Scope and terminology**
>
> This project recognizes two predefined static handshapes associated with selected ASL forms. It is not a general-purpose American Sign Language translator and does not recognize ASL grammar, sentences, facial expressions, or continuous signing.

## Project Purpose

The primary objective is to investigate how visual gestures can serve as an alternative input method for controlling simple electronic devices.

The project combines three technical areas:

- **Computer vision:** detecting a hand and extracting its spatial landmarks.
- **Machine learning:** classifying normalized landmark data into predefined command categories.
- **Embedded systems:** transmitting validated commands to an Arduino that controls physical outputs.

This approach provides a foundation for experimenting with gesture-based interfaces, assistive interaction concepts, and future device-control applications.

## How It Works

```text
User performs a predefined handshape
                |
                v
        Webcam frame capture
                |
                v
       MediaPipe HandLandmarker
          21 hand landmarks
                |
                v
       Landmark normalization
          63 numerical features
                |
                v
       Random Forest classifier
           ON / OFF prediction
                |
                v
      Confidence and stability checks
                |
                v
          Command mapping
       LIGHT_ON / LIGHT_OFF
                |
                v
       USB serial communication
             9600 baud
                |
                v
             Arduino
                |
                v
       Green LED / Red LED
```

## Key Features

- Real-time hand landmark detection using MediaPipe.
- Webcam-based input without requiring a dedicated gesture controller.
- Landmark normalization and 63-dimensional feature extraction.
- Random Forest classification for two predefined handshape classes.
- Confidence thresholds and command-stabilization logic.
- USB serial communication between Python and Arduino.
- Arduino firmware for controlling two low-voltage LED outputs.
- A simulation mode for development without connected hardware, if supported by the current application.
- A modular codebase separating vision, recognition, command processing, and hardware communication.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application logic and pipeline integration |
| OpenCV | Webcam access and frame processing |
| MediaPipe Tasks | Hand detection and 21-landmark extraction |
| NumPy | Numerical data processing |
| scikit-learn | Random Forest classification |
| Joblib | Saving and loading the trained model |
| PySerial | USB serial communication |
| Arduino C++ | Embedded command handling and LED control |

## System Architecture

The implementation is organized into independent modules so that individual components can be tested and improved without redesigning the entire application.

| Component | Responsibility |
|---|---|
| `vision/hand_detector.py` | Detects hands and extracts landmarks |
| `vision/preprocessing.py` | Normalizes landmark coordinates |
| `vision/feature_extractor.py` | Converts landmarks into a 63-value feature vector |
| `recognition/collect_data.py` | Collects labeled training samples |
| `recognition/train_classifier.py` | Trains and evaluates the classifier |
| `recognition/classifier.py` | Loads the model and predicts the handshape class |
| `control/command_parser.py` | Maps predictions to supported device commands |
| `control/debounce.py` | Prevents unstable or repeated command activation |
| `control/serial_controller.py` | Manages serial communication and connection state |
| `arduino/sign_controller/sign_controller.ino` | Processes commands and controls the LED outputs |
| `main.py` | Integrates the application pipeline and user interface |

## Recognition Pipeline

### 1. Hand Detection

MediaPipe identifies the visible hand and returns 21 landmarks representing key points on the fingers and palm.

The classifier operates on landmark-derived features rather than directly learning from raw webcam images.

### 2. Landmark Preprocessing

The detected landmarks are normalized to reduce sensitivity to variations in hand position and apparent size.

The preprocessing implementation is shared by data collection, model training, and real-time inference to keep feature generation consistent.

### 3. Feature Extraction

The 21 landmarks contain three coordinates each, producing a feature vector with 63 numerical values.

### 4. Classification

A Random Forest classifier predicts one of the supported classes: `ON` or `OFF`.

The confidence threshold helps reject uncertain predictions. Actual recognition performance depends on the collected dataset, camera conditions, and implementation.

### 5. Command Stabilization

A predicted class must satisfy the configured stability conditions before a command is transmitted. This reduces accidental activation caused by temporary misclassifications.

### 6. Device Communication

The application maps the accepted class to `LIGHT_ON` or `LIGHT_OFF` and sends the command through a USB serial connection to the Arduino.

## ASL Reference and Recognition Scope

The selected handshapes are informed by the following American Sign Language references from HandSpeak:

- [ON](https://www.handspeak.com/word/1553/)
- [OFF](https://www.handspeak.com/word/1536/)
- [LIGHT](https://www.handspeak.com/word/1272/)

The prototype uses selected fingerspelled handshapes associated with `#ON` and `#OFF`, holding the final handshape for classification.

This distinction matters because some sign forms are movement-based. A classifier operating on a single frame of hand landmarks cannot reliably distinguish signs whose meaning depends on movement alone.

The implementation therefore focuses on two visually distinguishable static poses rather than claiming to interpret the full linguistic meaning of ASL.

ASL is a complete natural language, and sign languages are not universal. The prototype does not automatically generalize to other sign languages or to all people who sign.

## Accessibility and Human–Computer Interaction

Gesture-based interaction can provide an additional input option in situations where voice commands, keyboards, or touch interfaces may be inconvenient or unsuitable.

This project is an early technical exploration of that possibility. It is not a validated assistive technology, and its current two-command vocabulary is intentionally limited.

Future work could investigate broader gesture vocabularies, movement-aware recognition, alternative input methods, and integration with additional devices.

## Hardware Requirements

- Arduino Uno or compatible board
- Webcam
- Two LEDs
- Two 220 Ω current-limiting resistors
- USB data cable
- Computer capable of running the Python application

### LED Wiring

```text
Arduino D8 ── 220 Ω resistor ── Green LED anode
Green LED cathode ── GND

Arduino D9 ── 220 Ω resistor ── Red LED anode
Red LED cathode ── GND
```

| Command | Green LED (D8) | Red LED (D9) |
|---|---|---|
| `LIGHT_ON` | ON | OFF |
| `LIGHT_OFF` | OFF | ON |

**Safety:** This prototype is designed for low-voltage LED outputs only. Never connect Arduino GPIO pins directly to mains voltage or other hazardous electrical loads.

## Installation

### Requirements

- Python 3.10 or a compatible version supported by the dependencies
- Git
- A webcam
- Arduino hardware for physical device control

Clone the repository:

```bash
git clone https://github.com/Ab40D/sign-language-recognition.git
cd sign-language-recognition
```

Create and activate a virtual environment.

**Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell:**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Download the MediaPipe hand landmark model if it is not already included:

```bash
python scripts/fetch_hand_landmarker.py
```

Upload the Arduino firmware from:

```text
arduino/sign_controller/sign_controller.ino
```

Use the Arduino IDE or another compatible tool to compile and upload the sketch.

## Collect Training Data

The classifier requires labeled samples for both supported classes.

From the repository root, run the collection script for each class:

```bash
python recognition/collect_data.py --label ON
python recognition/collect_data.py --label OFF
```

Follow the on-screen instructions and hold the corresponding handshape in front of the webcam.

Collect samples under different reasonable hand positions and camera distances. Consistent, representative samples are important for reducing misclassification.

The collected dataset is local training data; it is not a bundled public image dataset.

## Train the Classifier

After collecting samples for both classes, train the model:

```bash
python recognition/train_classifier.py
```

The training script is intended to load the collected samples, train the Random Forest classifier, report evaluation metrics, and save the model for inference.

Model evaluation results should be interpreted in the context of the dataset and split methodology. A high score on a small or highly similar dataset does not establish reliable performance across different users or environments.

## Run the Application

Start the application:

```bash
python main.py
```

If supported by the current version, the following options can be used for development and troubleshooting:

```bash
python main.py --port /dev/ttyACM0
python main.py --port COM3
python main.py --simulate
python main.py --list-ports
python main.py --send LIGHT_ON
```

The serial port depends on the operating system and the Arduino connection. Check the application's available options if a flag is not recognized.

## Testing

Run the project's unit and integration tests:

```bash
python -m unittest discover -s tests
```

Tests that use synthetic landmarks or simulated serial responses can verify parts of the software pipeline without requiring a webcam or connected Arduino. They do not, by themselves, establish real-world recognition accuracy or hardware reliability.

## Troubleshooting

| Issue | Suggested action |
|---|---|
| Hand landmark model missing | Run the model download script |
| Webcam unavailable | Check camera permissions and close applications using the camera |
| Arduino not detected | Check the USB data cable and list available serial ports |
| Serial permission denied on Linux | Check your user's serial-device permissions and system group configuration |
| No Arduino response | Verify the uploaded sketch and ensure the baud rate matches |
| Unstable recognition | Improve lighting, keep one hand visible, and collect more representative samples |
| Incorrect predictions | Review the dataset, retrain the classifier, and evaluate both classes |
| Repeated commands | Review the stability, cooldown, and duplicate-command settings |

## Limitations

- **Limited vocabulary:** Only two predefined classes are supported.
- **Static handshapes:** The current approach does not model complete gesture sequences or movement-dependent signs.
- **User and environment variation:** Recognition may change with hand anatomy, camera angle, lighting, and background conditions.
- **No linguistic interpretation:** The system does not translate ASL sentences or understand grammar, facial expressions, or conversational context.
- **Hardware scope:** The demonstrated output is limited to low-voltage LED control.
- **Accessibility validation:** The prototype has not been established as a validated assistive device.

## Future Development

Potential extensions include:

- Temporal models for movement-aware sign recognition.
- Additional command classes and expanded gesture vocabularies.
- More robust evaluation across different users and environments.
- Integration with MQTT and Home Assistant.
- Support for additional embedded devices and output interfaces.
- Exploration of gesture-driven smart-home and human–computer interaction applications.

These are future directions, not claims about features already implemented.


## Dataset and Data Collection

The project supports two approaches to collecting training data:

- **Webcam collection:** Capture labeled hand-landmark samples directly from a webcam using `recognition/collect_data.py`.
- **Video-based extraction:** Extract hand landmarks from labeled `ON` and `OFF` video recordings using `recognition/collect_from_video.py`.

The resulting samples are normalized landmark feature vectors rather than raw images used directly by the classifier. This keeps the recognition pipeline focused on hand geometry.

For reliable evaluation, training and test data should be separated carefully. When multiple recordings or participants are available, evaluate whether the model generalizes to recordings or people it has not seen during training.

Before redistributing any recordings, verify that their use and publication are authorized.

### Webcam collection

```bash
python recognition/collect_data.py --label ON
python recognition/collect_data.py --label OFF
```

### Extract samples from the bundled video directory

```bash
python recognition/collect_from_video.py
```

The video extraction script reads recordings from `data/videos/` by default and writes extracted samples to `data/samples/`. Review the script's options before changing the input or output directories.

### Train the classifier

```bash
python recognition/train_classifier.py
```

The training script reports held-out test metrics. These results describe the selected dataset and split; they should not be interpreted as proof of generalization to unseen users unless the evaluation explicitly tests that scenario.

## Project Background

This project evolves from an earlier voice-controlled Arduino LED prototype. The previous implementation is preserved in [`legacy_voice/`](legacy_voice/README.md), where applicable.

The main change is the input modality: webcam-based handshape recognition replaces voice input, while the project retains the general concept of mapping an input event to a serial command and an embedded-system response.

## Demonstration

A demonstration video is available on YouTube:

[Watch the project demonstration](https://www.youtube.com/watch?v=P_Yeaprbcxw&t=4s)

## Author

**Abdelkhalek Mammeri**

- GitHub: [@Ab40D](https://github.com/Ab40D)
- LinkedIn: [Abdelkhalek Mammeri](https://www.linkedin.com/in/abdelkhalekmammeri/)

## License

A license has not yet been specified. Until one is added to the repository, do not assume that the project is licensed for unrestricted reuse or redistribution.

---

*An experimental project exploring computer vision, machine learning, embedded systems, and gesture-based human–computer interaction.*
