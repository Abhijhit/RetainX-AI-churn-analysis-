const multer = require('multer');
const path = require('path');

// Configure storage
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    const uploadPath = process.env.UPLOAD_PATH || './uploads';
    cb(null, uploadPath);
  },
  filename: function (req, file, cb) {
    // Generate unique filename
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, file.fieldname + '-' + uniqueSuffix + path.extname(file.originalname));
  }
});

// File filter
const fileFilter = (req, file, cb) => {
  // Accept CSV and Excel files
  const allowedTypes = ['.csv', '.xlsx', '.xls'];
  const ext = path.extname(file.originalname).toLowerCase();
  
  if (allowedTypes.includes(ext)) {
    cb(null, true);
  } else {
    cb(new Error('Only CSV and Excel files are allowed'), false);
  }
};

// Create upload middleware
const upload = multer({
  storage: storage,
  fileFilter: fileFilter,
  limits: {
    fileSize: (process.env.MAX_FILE_SIZE || 10) * 1024 * 1024 // MB to bytes
  }
});

// Create uploads directory if it doesn't exist
const fs = require('fs');
const uploadPath = process.env.UPLOAD_PATH || './uploads';
if (!fs.existsSync(uploadPath)) {
  fs.mkdirSync(uploadPath, { recursive: true });
  console.log(`Created uploads directory: ${uploadPath}`);
}

module.exports = upload;