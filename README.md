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

And

```
cd opencv-text-recognition
node RELAY-API/app.js
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
