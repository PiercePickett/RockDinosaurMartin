/*
 * dino_client.ino
 * ===============
 * Arduino (ESP8266 or ESP32) client for the Jetson dino_server.py.
 *
 * What it does
 * ------------
 *  1. Connects to your WiFi network.
 *  2. Polls GET http://<JETSON_IP>:5000/classify every second.
 *  3. Parses the JSON response {"class":"...","confidence":0.xx,...}
 *  4. Drives a servo on pin SERVO_PIN based on the detected dino class.
 *  5. Optionally flashes an LED to indicate confidence.
 *
 * Hardware
 * --------
 *  - ESP8266 (NodeMCU / Wemos D1) *or* ESP32 — both work identically.
 *  - Servo on SERVO_PIN (default D5 / GPIO14).
 *  - Optional LED on LED_PIN (default D4 / GPIO2, which is the NodeMCU built-in).
 *
 * Libraries needed (install via Arduino Library Manager)
 * -------------------------------------------------------
 *  - ArduinoJson  (Benoit Blanchon) — tested with v6 & v7
 *  - ESP8266WiFi  (built-in for ESP8266 board package)
 *    *or* WiFi.h  (built-in for ESP32 board package)
 *  - ESP8266HTTPClient / HTTPClient (built-in for each package)
 *  - Servo  (built-in)
 *
 * Board packages
 * --------------
 *  ESP8266: https://arduino.esp8266.com/stable/package_esp8266com_index.json
 *  ESP32:   https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
 */

// ─── CONFIGURATION ────────────────────────────────────────────────────────────

#define WIFI_SSID     "Kent State Guest"
#define WIFI_PASS     ""

// IP address of your Jetson on the local network.
// Find it with:  hostname -I   (on the Jetson terminal)
#define JETSON_IP    "10.7.112.166"
#define JETSON_PORT   5000

// How often to ask the Jetson for a classification (milliseconds)
#define POLL_INTERVAL_MS  500

// Servo pin and angle map
#define SERVO_PIN     14   // D5 on NodeMCU

// LED pin — LOW = on for NodeMCU built-in LED
#define LED_PIN       2    // D4 on NodeMCU (active LOW)

// Confidence threshold — results below this are treated as "unknown"
#define MIN_CONFIDENCE  0.50f

// ─── CLASS → SERVO ANGLE MAP ──────────────────────────────────────────────────
// Edit these to match your class_names.json and desired servo positions.
// Unknown / low-confidence → 90° (center)
struct DinoAngle {
  const char* className;
  int angle;
};

const DinoAngle DINO_MAP[] = {
  { "red",    0   },
  { "blue",   45  },
  { "green",  90  },
  { "yellow", 135 },
  // Add more classes here as needed:
  // { "triceratops", 60 },
  // { "trex",        120 },
};
const int DINO_MAP_LEN = sizeof(DINO_MAP) / sizeof(DINO_MAP[0]);
const int UNKNOWN_ANGLE = 90;  // center — used when confidence is too low

// ─── PLATFORM INCLUDES ────────────────────────────────────────────────────────

#if defined(ESP8266)
  #include <ESP8266WiFi.h>
  #include <ESP8266HTTPClient.h>
  #include <WiFiClient.h>
#elif defined(ESP32)
  #include <WiFi.h>
  #include <HTTPClient.h>
#else
  #error "This sketch requires an ESP8266 or ESP32 board."
#endif

#include <ArduinoJson.h>
#include <Servo.h>

// ─── GLOBALS ──────────────────────────────────────────────────────────────────

Servo servo;
unsigned long lastPollMs = 0;
int currentAngle = UNKNOWN_ANGLE;

// ─── HELPERS ──────────────────────────────────────────────────────────────────

int angleForClass(const char* className) {
  for (int i = 0; i < DINO_MAP_LEN; i++) {
    if (strcasecmp(className, DINO_MAP[i].className) == 0) {
      return DINO_MAP[i].angle;
    }
  }
  return -1;  // not found
}

void setServoAngle(int angle) {
  if (angle == currentAngle) return;
  currentAngle = angle;
  servo.write(angle);
  Serial.print("Servo → ");
  Serial.print(angle);
  Serial.println("°");
}

void connectWiFi() {
  Serial.print("Connecting to WiFi: ");
  Serial.print(WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("Connected! IP: ");
  Serial.println(WiFi.localIP());
}

// ─── SETUP ────────────────────────────────────────────────────────────────────

void setup() {
  Serial.begin(115200);
  delay(100);
  Serial.println("\n=== dino_client starting ===");

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, HIGH);  // off (active LOW)

  servo.attach(SERVO_PIN);
  servo.write(UNKNOWN_ANGLE);

  connectWiFi();

  // Quick health check
  Serial.print("Checking Jetson health at http://");
  Serial.print(JETSON_IP);
  Serial.print(":");
  Serial.println(JETSON_PORT);

  WiFiClient client;
  HTTPClient http;
  String healthUrl = String("http://") + JETSON_IP + ":" + JETSON_PORT + "/health";
  http.begin(client, healthUrl);
  int code = http.GET();
  if (code == 200) {
    Serial.println("Jetson server is up ✓");
    Serial.println(http.getString());
  } else {
    Serial.print("Health check failed, HTTP code: ");
    Serial.println(code);
  }
  http.end();
}

// ─── LOOP ─────────────────────────────────────────────────────────────────────

void loop() {
  // Reconnect if WiFi drops
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi lost — reconnecting…");
    connectWiFi();
  }

  unsigned long now = millis();
  if (now - lastPollMs < POLL_INTERVAL_MS) return;
  lastPollMs = now;

  // ── GET /classify ──────────────────────────────────────────────────────────
  WiFiClient client;
  HTTPClient http;
  String url = String("http://") + JETSON_IP + ":" + JETSON_PORT + "/classify";
  http.begin(client, url);
  http.setTimeout(3000);  // 3 s timeout
  int code = http.GET();

  if (code != 200) {
    Serial.print("HTTP error: ");
    Serial.println(code);
    http.end();
    return;
  }

  String body = http.getString();
  http.end();

  // ── Parse JSON ─────────────────────────────────────────────────────────────
  // {"class":"triceratops","class_index":2,"confidence":0.91,"scores":{...},...}
  StaticJsonDocument<512> doc;
  DeserializationError err = deserializeJson(doc, body);
  if (err) {
    Serial.print("JSON parse error: ");
    Serial.println(err.c_str());
    return;
  }

  const char* dinoClass = doc["class"] | "unknown";
  float confidence = doc["confidence"] | 0.0f;

  Serial.print("Class: ");
  Serial.print(dinoClass);
  Serial.print("  Confidence: ");
  Serial.println(confidence, 3);

  // ── Drive servo ────────────────────────────────────────────────────────────
  int angle;
  if (confidence >= MIN_CONFIDENCE) {
    angle = angleForClass(dinoClass);
    if (angle < 0) {
      Serial.println("  (class not in map — using center)");
      angle = UNKNOWN_ANGLE;
    }
    // Flash LED briefly to signal a confident detection
    digitalWrite(LED_PIN, LOW);   // on
    delay(30);
    digitalWrite(LED_PIN, HIGH);  // off
  } else {
    Serial.println("  (confidence too low)");
    angle = UNKNOWN_ANGLE;
  }

  setServoAngle(angle);
}

// ─── Optional: POST your own image to the Jetson ─────────────────────────────
/*
  If you have an OV2640 camera module attached to your ESP32-CAM, you can
  capture a frame and POST it instead of having the Jetson use its own camera:

  #include "esp_camera.h"

  void classifyLocalFrame() {
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) return;

    WiFiClient client;
    HTTPClient http;
    http.begin(client, "http://" JETSON_IP ":" + String(JETSON_PORT) + "/classify");
    http.addHeader("Content-Type", "image/jpeg");
    int code = http.POST(fb->buf, fb->len);
    esp_camera_fb_return(fb);

    if (code == 200) {
      String body = http.getString();
      // ... parse JSON as above
    }
    http.end();
  }
*/
