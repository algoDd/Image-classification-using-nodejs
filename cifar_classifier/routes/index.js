const tf = require('@tensorflow/tfjs');
const tfNode = require('@tensorflow/tfjs-node');
const multer = require('multer');
const fs = require('fs')
const path = require('path')
const filePath = path.join(__dirname, 'uploads/images/photo.jpg');
const class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

const storage = multer.diskStorage({
  destination: __dirname + '/uploads/images',
  filename: function (req, file, cb) {
    cb(null, file.fieldname + ".jpg")
  }
})
const upload = multer({
  storage: storage
});


var express = require('express');
var router = express.Router();

router.get('/', function (req, res, next) {
  res.render('index', {
    title: 'Express'
  });
});

router.post('/upload', upload.single('photo'), (req, res) => {
  if (req.file) {
    res.redirect('/predict');
  } else throw 'error';
});

router.get('/predict', async (req, res) => {
  const model = await tf.loadLayersModel('file://model/tfjs_model/model.json');
  let image_class;
  fs.readFile(filePath, (err, imageBuffer) => {
    const tfImage = tfNode.node.decodeImage(imageBuffer).resizeBilinear([32, 32]);
    console.log('Predict image');
    const predict = model.predict(tfImage.reshape([1, 32, 32, 3]));
    image_class = tf.argMax(predict.dataSync());
    console.log('Class: ', class_names[image_class.dataSync()]);
    fs.writeFile(path.join(__dirname, '../public/images/photo.jpg'), imageBuffer, function (err) {
      if (err) throw err;
      console.log('Updated!');
    });
    res.render('predict', {
      prediction: class_names[image_class.dataSync()]
    });
  });
});

module.exports = router;