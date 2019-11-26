# OpenCv Text Recognition Tunisian Carte Identity

this project create for give you service Api Rest with nodeJS for the downloading of photos of the Tunisian identity card in the first layer.
in the second layer, collect data from the images.
In the first step, the application uses Tesseract OCR to get the ID card number and save it in the outputbase.txt file.
in the second step the application uses the card number and creates a file named by that number
Finally, the application uses the text_recognaition.py python file to minimize the size of the image, change the color to black and white and get the image of the person in the identity card with OpenCv.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 
See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

in your machine you need :
```
npm
Tesseract OCR
Python 3
```

in python project you need :

```
OpenCV
imutils
numpy
pytesseract
argparse
```

in nodeJs part you need :
```    
"body-parser": "^1.18.3",
"cookie-parser": "~1.4.3",
"debug": "~2.6.9",
"express": "~4.16.0",
"faker": "^4.1.0",
"http-errors": "~1.6.2",
"morgan": "~1.9.0",
"multer": "^1.4.1",
"n-readlines": "^1.0.0",
"nodemon": "^1.18.9",
"pug": "2.0.0-beta11"
```


### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
git clone https://github.com/wajdibenhelal/opencv-text-recognition.git
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

```
cd opencv-text-recognition
node RELAY-API/app.js
```

### Break down into end to end tests

Explain what these tests test and why

for upload the identity card :
```
http://localhost:3003/upload
  =>profile  => PicPath
```
for open the identity card from the server :
```
http://localhost:3003/upload/picName.jpg
```
for cheack your data from the server :
```
cd PROJECT_PATH
cat NumberCard.txt
```
## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
