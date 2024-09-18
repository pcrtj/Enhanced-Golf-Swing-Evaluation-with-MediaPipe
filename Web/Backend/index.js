const express = require('express');
const mysql = require('mysql2');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
const jwt = require('jsonwebtoken');
const multer = require('multer');
const { spawn } = require('child_process');
const app = express();

app.use(cors());
app.use(express.json({ limit: '500mb' }));

const db = mysql.createConnection({
    user: 'root',
    host: 'localhost',
    password: '1234',
    database: 'golfswing_db'
});

db.connect(err => {
    if (err) {
        console.error('Error connecting to the database:', err);
        process.exit(1);
    }
    console.log('Connected to the database successfully');
});

const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    const { username } = req.body;
    db.query('SELECT U_ID FROM `user` WHERE U_Username = ?', [username], (error, results) => {
      if (error) {
        cb(error, null);
      } else if (results.length === 0) {
        cb(new Error('User not found'), null);
      } else {
        const U_ID = results[0].U_ID;
        const dir = `./uploads/${U_ID}`;
        fs.mkdir(dir, { recursive: true }, (err) => {
          if (err) {
            cb(err, null);
          } else {
            cb(null, dir);
          }
        });
      }
    });
  },
  filename: function (req, file, cb) {
    cb(null, Date.now() + path.extname(file.originalname));
  }
});

const upload = multer({ storage: storage });

// Serve static files from the uploads directory
app.use('/uploads', express.static('uploads'));

// Routes
app.get('/', (req, res) => {
    res.send("Welcome to the Golf Swing Analysis API");
});

app.post('/login', (req, res) => {
    const username = req.body.username;
    const password = req.body.password;

    db.query('SELECT * FROM `user` WHERE U_Username=?', [username], (error, results, fields) => {
        if (error) {
            console.error('Error executing SQL query:', error);
            res.status(500).send('Internal Server Error');
        } else {
            if (results.length > 0) {
                console.log(results);
                if (results[0].U_Password === password) {
                    return res.status(201).send('Success');
                } else {
                    return res.status(200).send('Fail');
                }
            } else {
                console.log('No user found with the provided username.');
                res.status(202).send('User not found');
            }
        }
    });
});



app.post('/signup', (req, res) => {
    const { username, password } = req.body;
    
    db.query('SELECT * FROM `user` WHERE U_Username=?', [username], (error, results) => {
        if (error) {
            console.error('Error executing SQL query:', error);
            return res.status(500).send('Internal Server Error');
        }

        if (results.length > 0) {
            return res.status(200).send('user already exist');
        } else {
            // เพิ่มผู้ใช้ใหม่ในตาราง user
            db.query(
                'INSERT INTO `user` (U_Username, U_Password ) VALUES (?, ?)',
                [username, password],
                (err, result) => {
                    if (err) {
                        console.error('Error executing SQL query:', err);
                        return res.status(500).send('Internal Server Error');
                    }
                    return res.status(201).send('Success');
                }
            );
        }
    });
});

app.post('/upload', upload.single('video'), (req, res) => {
    if (!req.file) {
        return res.status(400).send('No file uploaded.');
    }

    const { username, startTime, endTime } = req.body;
    const inputVideoPath = req.file.path;

    db.query('SELECT U_ID FROM `user` WHERE U_Username = ?', [username], (error, results) => {
        if (error) {
            console.error('Error querying database:', error);
            return res.status(500).send('Internal Server Error');
        }

        if (results.length === 0) {
            return res.status(404).send('User not found');
        }

        const U_ID = results[0].U_ID;

        // Call Python script to process video
        const pythonProcess = spawn('python', ['video_processor.py', inputVideoPath, startTime, endTime]);

        let pythonData = '';
        pythonProcess.stdout.on('data', (data) => {
            pythonData += data.toString();
        });

        pythonProcess.on('close', (code) => {
            if (code !== 0) {
                return res.status(500).send('Error processing video');
            }

            const processedData = JSON.parse(pythonData);

            // Insert event and get E_ID
            const { accuracy, avgAccuracy } = processedData;
            const sql = 'INSERT INTO event (U_ID, EAccuracy, EAvgAccuracy, EDateTime) VALUES (?, ?, ?, NOW())';
            db.query(sql, [U_ID, JSON.stringify(accuracy), avgAccuracy], (error, result) => {
                if (error) {
                    console.error('Error saving to database:', error);
                    return res.status(500).send('Error saving results');
                }

                const E_ID = result.insertId;

                // Create E_ID folder
                const eventDir = path.join(`./uploads/${U_ID}`, E_ID.toString());
                fs.mkdirSync(eventDir, { recursive: true });

                // Move input video to E_ID folder
                const newInputPath = path.join(eventDir, 'input' + path.extname(inputVideoPath));
                fs.renameSync(inputVideoPath, newInputPath);

                // Assume the Python script saves the processed video and returns its path
                const outputVideoPath = processedData.outputVideoPath;
                const newOutputPath = path.join(eventDir, 'output' + path.extname(outputVideoPath));
                fs.renameSync(outputVideoPath, newOutputPath);

                // Update event with video paths
                const updateSql = 'UPDATE event SET EVideoBefore = ?, EVideoAfter = ? WHERE E_ID = ?';
                db.query(updateSql, [newInputPath, newOutputPath, E_ID], (updateError) => {
                    if (updateError) {
                        console.error('Error updating event:', updateError);
                        return res.status(500).send('Error updating event');
                    }

                    res.json({
                        ...processedData,
                        E_ID,
                        inputVideoPath: newInputPath,
                        outputVideoPath: newOutputPath
                    });
                });
            });
        });
    });
});

app.get('/history', (req, res) => {
    const username = req.query.username;
    const page = parseInt(req.query.page) || 1;
    const rowsPerPage = 4; // Adjust this value as needed

    console.log(`Fetching history for username: ${username}, page: ${page}`);

    if (!username) {
        console.error('Username not provided');
        return res.status(400).json({ error: 'Username is required' });
    }

    db.query('SELECT U_ID FROM `user` WHERE U_Username = ?', [username], (error, results) => {
        if (error) {
            console.error('Error querying user:', error);
            return res.status(500).json({ error: 'Internal Server Error' });
        }

        if (results.length === 0) {
            console.error(`User not found: ${username}`);
            return res.status(404).json({ error: 'User not found' });
        }

        const U_ID = results[0].U_ID;
        console.log(`Found U_ID: ${U_ID} for username: ${username}`);

        const sql = `
            SELECT 
                E_ID,
                E_DateTime AS date,
                E_VideoBefore AS inputClip,
                E_VideoAfter AS outputClip,
                E_Score AS accuracy,
                E_AvgScore AS avgAccuracy
            FROM 
                event
            WHERE
                U_ID = ?
            ORDER BY 
                E_DateTime DESC
            LIMIT ? OFFSET ?;
        `;

        const offset = (page - 1) * rowsPerPage;

        db.query(sql, [U_ID, rowsPerPage, offset], (err, results) => {
            if (err) {
                console.error('Error fetching history data:', err);
                return res.status(500).json({ error: 'Error fetching history data', details: err.message });
            }

            console.log(`Raw results:`, results);

            try {
                // Process the results to parse the accuracy JSON string if necessary
                const processedResults = results.map(result => ({
                    ...result,
                    accuracy: typeof result.accuracy === 'string' ? JSON.parse(result.accuracy) : result.accuracy,
                    date: new Date(result.date).toLocaleString() // Format the date
                }));

                console.log(`Processed results:`, processedResults);

                res.setHeader('Content-Type', 'application/json');
                res.status(200).json(processedResults);
            } catch (parseError) {
                console.error('Error processing results:', parseError);
                res.status(500).json({ error: 'Error processing history data', details: parseError.message });
            }
        });
    });
});

const port = 3000;
app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
});