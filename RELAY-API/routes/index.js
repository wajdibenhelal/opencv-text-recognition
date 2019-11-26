var express = require('express');
var mysql = require('mysql');
var multer = require('multer');

var router = express.Router();


/* GET home page. */
router.get('/', function (req, res, next) {
    res.render('index', {title: 'Express'});
});


var storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, '/Volumes/SSD/opencv-text-recognition/RELAY-API/public/uploads')
    },
    filename: (req, file, cb) => {
        cb(null, file.fieldname + '-' + Date.now() + '.' + file.originalname)
    }
});
var upload = multer({storage: storage});

module.exports = upload;

router.post('/upload', upload.single('profile'), function (req, res) {
    if (!req.file) {
        console.log("No file received");
        message = "Error! in image upload.";
        res.render('index', {message: message, status: 'danger'});

    } else {
        var exec = require('child_process').exec;
        var cmd = 'python  /Volumes/SSD/opencv-text-recognition/text_recognition.py --east frozen_east_text_detection.pb    --image RELAY-API/public/uploads/' + req.file.filename + ' --padding 0.25 ';

        exec(cmd, function (error, stdout, stderr) {
            //console.log(stdout)
            var exec = require('child_process').exec;
            var cmd = 'tesseract images/Wn.jpg outputbase digits --psm 3';
            exec(cmd, function (error, stdout, stderr) {
                //console.log(stdout)
                var lineReader = require('readline').createInterface({
                    input: require('fs').createReadStream('/Volumes/SSD/opencv-text-recognition/outputbase.txt')
                });
                var cc;
                lineReader.on('line', function (line) {
                    console.log('Line from file:', line);
                    if (line.length == 8) {
                        console.log("ism ", line);
                        cc = line;
                        var exec = require('child_process').exec;
                        console.log(line);
                        var cmd = 'tesseract images/Wn.jpg ' + line + ' -l ara2+ara --psm 3';
                        line = null;
                        exec(cmd, function (error, stdout, stderr) {

                            console.log(stdout)
                        });
                    }

                });

                console.log('file received');
                console.log(req.file.filename);
                message = "Successfully! uploaded";
                res.json({image: req.file.filename});

            });
        });
//tesseract images/Wn.jpg outputbase -l ara2+ara --psm 3
    }
});


router.get('/upload/:cin', (req, res) => {

    /*var lineReader = require('readline').createInterface({
      input: require('fs').createReadStream('/Volumes/SSD/opencv-text-recognition/'+[req.params.cin]+'.txt')
    });
    var cc ;
    try {

    }catch (e) {
      console.log()
    }
    lineReader.on('line', function (line) {
      console.log('Line from file:', line);
    });

    //tesseract images/Wn.jpg outputbase -l ara2+ara --psm 3
        console.log('file received');
        message = "Successfully! uploaded";
        res.json({data: req.params.cin});
    */
    const fs = require('fs')

    const path = '/Volumes/SSD/opencv-text-recognition/' + [req.params.cin] + '.txt'

    try {
        if (fs.existsSync(path)) {
            console.log("existe");

            res.json({data: "true"});

        } else {
            res.json({data: "false"});
        }
    } catch (err) {
        res.json({data: "false"});
        console.error(err);

    }

});


module.exports = router;

