#include <TinyGPS++.h>
#include <HardwareSerial.h>

TinyGPSPlus gps;
HardwareSerial gpsSerial(2);
String nmeaSentence = "";

void setup() {
  Serial.begin(115200);
  gpsSerial.begin(9600, SERIAL_8N1, 16, 17);  
  Serial.println("GPS with TinyGPS++ and NMEA Sentences Starting...");
}

void loop() {
  while (gpsSerial.available() > 0) {
    char c = gpsSerial.read();
    
    // Build NMEA sentence character by character
    if (c == '$') {
      nmeaSentence = "$";  // Start of new sentence
    } else if (c == '\r' || c == '\n') {
      // End of sentence - print it if it's complete
      if (nmeaSentence.length() > 6 && nmeaSentence.startsWith("$")) {
        Serial.println("NMEA: " + nmeaSentence);
      }
      nmeaSentence = "";
    } else {
      nmeaSentence += c;
    }
    
    // Also feed character to TinyGPS++ for parsing
    if (gps.encode(c)) {
      // A complete, valid sentence was just processed
      displayParsedData();
    }
  }
}

void displayParsedData() {
  Serial.println("--- Parsed GPS Data ---");
  
  if (gps.location.isValid()) {
    Serial.print("Latitude: "); 
    Serial.println(gps.location.lat(), 6);
    Serial.print("Longitude: "); 
    Serial.println(gps.location.lng(), 6);
  } else {
    Serial.println("Location: Invalid");
  }
  
  if (gps.date.isValid()) {
    Serial.printf("Date: %02d/%02d/%04d\n", 
                  gps.date.month(), gps.date.day(), gps.date.year());
  }
  
  if (gps.time.isValid()) {
    // Convert UTC to local time (WITA = UTC+8)
    int localHour = (gps.time.hour() + 8) % 24;
    Serial.printf("Time UTC: %02d:%02d:%02d\n", 
                  gps.time.hour(), gps.time.minute(), gps.time.second());
    Serial.printf("Time Local (WITA): %02d:%02d:%02d\n", 
                  localHour, gps.time.minute(), gps.time.second());
  }
  
  if (gps.altitude.isValid()) {
    Serial.print("Altitude: "); 
    Serial.print(gps.altitude.meters());
    Serial.println(" meters");
  }
  
  if (gps.speed.isValid()) {
    Serial.print("Speed: "); 
    Serial.print(gps.speed.kmph());
    Serial.println(" km/h");
  }
  
  if (gps.course.isValid()) {
    Serial.print("Course: "); 
    Serial.print(gps.course.deg());
    Serial.println(" degrees");
  }
  
  Serial.print("Satellites: "); 
  Serial.println(gps.satellites.value());
  
  if (gps.hdop.isValid()) {
    Serial.print("HDOP: "); 
    Serial.println(gps.hdop.hdop());
  }
  
  Serial.println("========================");
  delay(2000);
}