/*
  voice_light_controller.ino
  Version 1 — USB serial light controller for Arduino Uno R3

  The PC understands speech. This sketch does not. It reads one ASCII
  line from USB serial and drives an LED.

  Protocol (keep this identical to docs/protocol.md):

    PC -> Uno     LIGHT_ON | LIGHT_OFF | STATUS
    Uno -> PC     READY
                  LIGHT_STATE:ON
                  LIGHT_STATE:OFF
                  ERROR:UNKNOWN_COMMAND
                  ERROR:LINE_TOO_LONG

  Every message is one line. This sketch's replies end with CR+LF
  because Serial.println() writes both. The PC strips that.

  Wiring, built-in LED (start here):
    No extra parts. LED_PIN stays LED_BUILTIN (digital pin 13).
    Watch the LED marked L. Ignore ON, TX, and RX.

  Wiring, optional external LED:
    Set LED_PIN to 8.
    D8 -> 220 ohm resistor -> LED anode (long leg)
    LED cathode (short leg, flat side) -> GND
    Do not use D0 or D1. Those pins are the USB serial UART.

  Board: Arduino Uno (ATmega328P)
  Baud:  9600 8N1
*/

#include <Arduino.h>
#include <string.h>

// LED_BUILTIN is pin 13 on the Uno R3, the onboard L LED.
// Change this to 8 only after wiring the external LED.
const int LED_PIN = LED_BUILTIN;

const long BAUD_RATE = 9600;

// Longest legal command is STATUS / LIGHT_OFF. 32 bytes is plenty
// and keeps this reader off the heap. The Uno has 2 KB of SRAM.
const size_t BUFFER_SIZE = 32;

char buffer[BUFFER_SIZE];
size_t bufferLength = 0;
bool overflow = false;

// Software copy of the last command. STATUS reports this instead of
// digitalRead(LED_PIN): on pin 13 the onboard LED circuit makes a
// raw read a worse source of truth than the state we commanded.
bool ledIsOn = false;

void setLed(bool on) {
  ledIsOn = on;
  digitalWrite(LED_PIN, on ? HIGH : LOW);
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
    setLed(true);
    Serial.println("LIGHT_STATE:ON");
  } else if (strcmp(command, "LIGHT_OFF") == 0) {
    setLed(false);
    Serial.println("LIGHT_STATE:OFF");
  } else if (strcmp(command, "STATUS") == 0) {
    Serial.println(ledIsOn ? "LIGHT_STATE:ON" : "LIGHT_STATE:OFF");
  } else {
    // Do not change the LED. A bad line must not look like a command.
    Serial.println("ERROR:UNKNOWN_COMMAND");
  }
}

void setup() {
  pinMode(LED_PIN, OUTPUT);
  setLed(false);

  Serial.begin(BAUD_RATE);

  // Do not write while (!Serial) here. That pattern is for native-USB
  // boards such as the Leonardo. On the Uno R3 it can hang forever.
  // Serial is usable as soon as begin() returns.

  // Opening the port from the PC resets the Uno (DTR -> RESET). The
  // PC starts reading immediately and waits for this line. A short
  // delay gives the USB host a moment to be listening; the PC-side
  // timeout covers the bootloader, which takes longer than this.
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
