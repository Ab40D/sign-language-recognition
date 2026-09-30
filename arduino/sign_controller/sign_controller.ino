/*
  sign_controller.ino
  ASL device-control prototype — USB serial light controller, Version 2

  Adapted from legacy_voice/voice_light_controller.ino (Version 1). The
  line reader, trimming, uppercasing, overflow handling, and protocol are
  the same battle-tested code; only the LED behaviour changed:

    V1: one LED on pin 13/8
    V2: two LEDs — D8 GREEN means ON, D9 RED means OFF

  The PC understands hand signs. This sketch does not. It reads one ASCII
  line from USB serial and drives the LEDs.

  Protocol (identical to Version 1, so the PC side stays compatible):

    PC -> Uno     LIGHT_ON | LIGHT_OFF | STATUS
    Uno -> PC     READY
                  LIGHT_STATE:ON
                  LIGHT_STATE:OFF
                  ERROR:UNKNOWN_COMMAND
                  ERROR:LINE_TOO_LONG

  LED behaviour:

    LIGHT_ON   -> green (D8) ON,  red (D9) OFF   -> reply LIGHT_STATE:ON
    LIGHT_OFF  -> green (D8) OFF, red (D9) ON    -> reply LIGHT_STATE:OFF
    STATUS     -> no change                       -> reply current state

  Every message is one line. Replies end with CR+LF because
  Serial.println() writes both; the PC strips that.

  Wiring (always use current-limiting resistors):

    D8 -> 220 ohm -> GREEN LED anode (long leg)  ; cathode -> GND
    D9 -> 220 ohm -> RED   LED anode (long leg)  ; cathode -> GND

  Safety: this project only switches low-voltage LEDs. Never connect
  Arduino GPIO to mains voltage.

  Board: Arduino Uno (ATmega328P)   Baud: 9600 8N1
*/

#include <Arduino.h>
#include <string.h>

const int GREEN_LED_PIN = 8;   // green = device ON
const int RED_LED_PIN = 9;     // red   = device OFF

const long BAUD_RATE = 9600;

// Longest legal command is LIGHT_OFF. 32 bytes is plenty and keeps this
// reader off the heap. The Uno has 2 KB of SRAM.
const size_t BUFFER_SIZE = 32;

char buffer[BUFFER_SIZE];
size_t bufferLength = 0;
bool overflow = false;

// Software copy of the state we commanded. STATUS reports this; it is the
// source of truth for what the LEDs should be doing.
bool deviceIsOn = false;

void applyState(bool on) {
  deviceIsOn = on;
  digitalWrite(GREEN_LED_PIN, on ? HIGH : LOW);
  digitalWrite(RED_LED_PIN, on ? LOW : HIGH);  // red is the inverse of green
}

void trimInPlace(char *text) {
  char *start = text;
  while (*start == ' ' || *start == '\t' || *start == '\r') {
    start++;
  }
  if (start != text) {
    memmove(text, start, strlen(start) + 1);
  }

  size_t length = strlen(text);
  while (length > 0 &&
         (text[length - 1] == ' ' || text[length - 1] == '\t' ||
          text[length - 1] == '\r')) {
    text[--length] = '\0';
  }
}

void toUpperInPlace(char *text) {
  for (char *cursor = text; *cursor != '\0'; ++cursor) {
    if (*cursor >= 'a' && *cursor <= 'z') {
      *cursor = *cursor - ('a' - 'A');
    }
  }
}

void handleCommand(char *command) {
  trimInPlace(command);
  toUpperInPlace(command);

  // Serial Monitor sometimes sends a blank line. Ignore it.
  if (command[0] == '\0') {
    return;
  }

  if (strcmp(command, "LIGHT_ON") == 0) {
    applyState(true);
    Serial.println("LIGHT_STATE:ON");
  } else if (strcmp(command, "LIGHT_OFF") == 0) {
    applyState(false);
    Serial.println("LIGHT_STATE:OFF");
  } else if (strcmp(command, "STATUS") == 0) {
    Serial.println(deviceIsOn ? "LIGHT_STATE:ON" : "LIGHT_STATE:OFF");
  } else {
    // Do not touch the LEDs. A bad line must not look like a command.
    Serial.println("ERROR:UNKNOWN_COMMAND");
  }
}

void setup() {
  pinMode(GREEN_LED_PIN, OUTPUT);
  pinMode(RED_LED_PIN, OUTPUT);
  applyState(false);  // device starts OFF -> red lit, green dark

  Serial.begin(BAUD_RATE);

  // Do not write while (!Serial) here. That pattern is for native-USB
  // boards such as the Leonardo; on the Uno R3 it can hang forever.

  // Opening the port from the PC resets the Uno (DTR -> RESET). The PC
  // waits for this READY line; the short delay gives the USB host a
  // moment to start listening (the PC-side timeout covers the rest).
  delay(50);
  Serial.println("READY");
}

void loop() {
  while (Serial.available() > 0) {
    int raw = Serial.read();
    if (raw < 0) {
      break;
    }

    char incoming = (char)raw;
    if (incoming == '\n') {
      buffer[bufferLength] = '\0';
      if (overflow) {
        Serial.println("ERROR:LINE_TOO_LONG");
      } else {
        handleCommand(buffer);
      }
      bufferLength = 0;
      overflow = false;
    } else if (incoming == '\r') {
      // Hosts that send CRLF would otherwise glue '\r' onto the token.
    } else if (bufferLength < BUFFER_SIZE - 1) {
      buffer[bufferLength++] = incoming;
    } else {
      overflow = true;
    }
  }
}
