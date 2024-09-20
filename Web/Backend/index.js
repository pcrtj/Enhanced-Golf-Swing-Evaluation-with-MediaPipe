const express = require('express');
const mysql = require('mysql2');
const cors = require('cors');
const multer = require('multer');
const axios = require('axios');
const app = express();
const fs = require('fs').promises; // Use promises version of fs
const path = require('path');

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
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

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

app.post('/upload', (req, res) => {
    const username = req.body.username;
    const video = req.file; // ตรวจสอบว่ามีไฟล์ถูกส่งมา

    // ตรวจสอบว่ามี username และไฟล์วิดีโอถูกส่งมาหรือไม่
    if (!username || !video) {
        return res.status(400).json({ message: "Username and video are required" });
    }

    // ตรวจสอบว่าพบผู้ใช้หรือไม่
    const query = 'SELECT * FROM users WHERE username = ?';
    connection.query(query, [username], (error, results) => {
        if (error) {
            console.error("Database error: ", error);
            return res.status(500).json({ message: "Database error" });
        }

        if (results.length === 0) {
            return res.status(404).json({ message: "User not found" });
        }

        // ดำเนินการต่อหลังจากพบผู้ใช้และไฟล์ถูกต้อง
        // ...
    });
});

app.post('/save-result', async (req, res) => {
    try {
        const { username, temp_folder, input_video, output_video, similarities, average_similarity } = req.body;

        // Get U_ID from username
        const [userResults] = await db.promise().query('SELECT U_ID FROM `user` WHERE U_Username = ?', [username]);
        
        if (userResults.length === 0) {
            return res.status(404).json({ error: 'User not found' });
        }

        const U_ID = userResults[0].U_ID;

        // Insert into event table and get the auto-generated E_ID
        const [result] = await db.promise().query(
            'INSERT INTO event (U_ID, E_DateTime, E_VideoBefore, E_VideoAfter, E_Score, E_AvgScore) VALUES (?, NOW(), ?, ?, ?, ?)',
            [U_ID, input_video, output_video, JSON.stringify(similarities), average_similarity]
        );

        const E_ID = result.insertId;

        // Create new folder with E_ID
        const eventFolder = path.join(__dirname, 'uploads', E_ID.toString());
        await fs.mkdir(eventFolder, { recursive: true });

        // Move files from temp folder to E_ID folder
        const filesToMove = ['input_golf.mp4', 'output_golf.mp4', 'data.csv', 'predicted_data.csv', 'adjusted_golf.mp4'];
        for (const file of filesToMove) {
            const oldPath = path.join(temp_folder, file);
            const newPath = path.join(eventFolder, file);
            if (await fs.stat(oldPath).catch(() => false)) {
                await fs.rename(oldPath, newPath);
            }
        }

        // Remove temp folder
        await fs.rm(temp_folder, { recursive: true, force: true }).catch(console.error);

        // Update file paths in database
        await db.promise().query(
            'UPDATE event SET E_VideoBefore = ?, E_VideoAfter = ? WHERE E_ID = ?',
            [path.join(E_ID.toString(), 'input_golf.mp4'), path.join(E_ID.toString(), 'output_golf.mp4'), E_ID]
        );

        res.status(200).json({ message: 'Event saved successfully', E_ID: E_ID });
    } catch (error) {
        console.error('Error in save-result:', error);
        res.status(500).json({ error: 'Internal Server Error', details: error.message });
    }
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