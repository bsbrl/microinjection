                     /*
  Blink

  Turns an LED on for one second, then off for one second, repeatedly.

  Most Arduinos have an on-board LED you can control. On the UNO, MEGA and ZERO
  it is attached to digital pin 13, on MKR1000 on pin 6. LED_BUILTIN is set to
  the correct LED pin independent of which board is used.
  If you want to know what pin the on-board LED is connected to on your Arduino
  model, check the Technical Specs of your board at:
  https://www.arduino.cc/en/Main/Products

  modified 8 May 2014
  by Scott Fitzgerald
  modified 2 Sep 2016
  by Arturo Guadalupi
  modified 8 Sep 2016
  by Colby Newman

  This example code is in the public domain.

  http://www.arduino.cc/en/Tutorial/Blink
*/

String readString; 
String data, Light, Pressure;
int L, l, P, p, PSI;
int t;
int flag;
int input;
int L1_pin = 7;
int L2_pin = 8;
//int P_pin = 11;

void setup() //set up the arduino
{
  // initialize digital pin LED_BUILTIN as an output.
  Serial.begin(9600); //frequency
  //pinMode(P_pin, OUTPUT); //one of the output
  pinMode(L1_pin, OUTPUT); //another output Output:1-11
  pinMode(L2_pin, OUTPUT);
}

// the loop function runs over and over again forever
void loop()  //run again and again
{
  while (!Serial.available()) //coming from CPU !Serical.available if signal is available
  {
    if (flag == 0)
    {
      digitalWrite(L1_pin, LOW);
      digitalWrite(L2_pin, LOW);
    }
    if (flag == 1) //Lpin:7.  High:5.   Low:0
    {
      digitalWrite(L1_pin, HIGH);//digitalWrite:make the port be high or low
      digitalWrite(L2_pin,LOW);
    }
    if (flag == 2) //Lpin:7.  High:5.   Low:0
    {
      digitalWrite(L1_pin, LOW);//digitalWrite:make the port be high or low
      digitalWrite(L2_pin,HIGH);
    }
    if (flag == 3) //Lpin:7.  High:5.   Low:0
    {
      digitalWrite(L1_pin, HIGH);//digitalWrite:make the port be high or low
      digitalWrite(L2_pin,HIGH);
    }
    delay(30);//30 ms taken to change
    /*
    Serial.print("Flag = ");
    Serial.println(flag);
    */
  }
  
  while (Serial.available() > 0) //if there is a signal coming, Serial.available would read the signal letter by letter automatically
  {
    char c = Serial.read();//L1l, read this signal as a string one digit by one digit
    readString += c;//
  }
  
  if (readString.length() > 0)//if readstring has something in it
  {
    Serial.print("Arduino Received:");
    Serial.println(readString);//print the signal
    L = readString.indexOf("L"); // 0
    l = readString.indexOf("l"); // 2
    //P = readString.indexOf("P"); //to adjust the pressure
    //p = readString.indexOf("p");
    if (L >= 0)
    {
      Light = readString.substring(L+1,l); //L+1:1 l:2
      t = Light.toInt();//convert string to integer
      Serial.print(t);
      if (t == 0)
      {
        digitalWrite(L1_pin, LOW);
        //delay(100);
        digitalWrite(L2_pin,LOW);
        delay(100);
        flag = 0;
      }
      if (t == 1)
      {
        digitalWrite(L1_pin, HIGH);
        //delay(100);
        digitalWrite(L2_pin,LOW);
        delay(100);
        flag = 1;
        //Serial.println(flag);
      }
      if (t == 2)
      {
        digitalWrite(L1_pin, LOW);
        //delay(100);
        digitalWrite(L2_pin,HIGH);
        delay(100);
        flag = 2;
      }
      if (t == 3)
      {
        digitalWrite(L1_pin, HIGH);
        //delay(100);
        digitalWrite(L2_pin,HIGH);
        delay(100);
        flag = 3;
      }
      Serial.print("T value:");
      Serial.println(t);
    }
    readString = "";
   /* if (P >= 0)
    {
      Pressure = readString.substring(P+1,p);
      PSI = Pressure.toInt();
      analogWrite(P_pin, PSI);
      delay(500);
      if (PSI == 0)
      {
        flag = 0;
      }
      else 
      {
        flag = 1;
      }
    }
    readString = "";
    /*
    input = readString.toInt();
    Serial.println(input);
    if (input >= 0)
    {
      analogWrite(11, input);
      delay(500);
    }
    readString = "";
    */
  }
}
