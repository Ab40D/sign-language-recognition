# Legacy voice-controlled version (preserved, no longer the main project)

This folder contains **Version 1** of the project: the voice-controlled
Arduino LED. The main project has moved to ASL hand-sign control (see the
repository root), and the microphone/speech input was retired.

Nothing here is imported by the new code. It is kept because:

* `voice_light_controller.ino` is the direct ancestor of
  `arduino/sign_controller/sign_controller.ino` (same serial protocol).
* `voice_controller.py` contains the original `ArduinoLink` serial class,
  port auto-detection, and error messages that the new
  `control/serial_controller.py` reuses.
* `voice_command_parser.py` shows the phrase->command mapping idea that
  `control/command_parser.py` continues (sign -> command instead of
  speech -> command).

If you want to run the voice version again, its original README, wiring
diagram, architecture and flowchart images live here too. Dependencies
(pyserial, SpeechRecognition) are NOT in the main requirements.txt anymore.

Run its parser tests with:

    cd legacy_voice && python3 test_voice_parser.py
