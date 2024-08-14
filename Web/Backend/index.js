const express = require('express');
const mysql = require('mysql2');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
const jwt = require('jsonwebtoken');
const Swal = require('sweetalert2')
const app = express();

app.use(cors());
app.use(express.json({ limit: '500mb' }));

// Use environment variables from the .env file
const db = mysql.createConnection({
    user: 'root',
    host: 'localhost',
    password: '1234',
    database: 'golfswing_db'
});

// Connect to the database
db.connect(err => {
    if (err) {
        console.error('Error connecting to the database:', err);
        process.exit(1); // Exit the application if database connection fails
    }
    console.log('Connected to the database successfully');
});


app.get('/', (req, res) => {
    res.send("still");
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
                    
                    res.send("Success");
                } else {
                    
                    res.send("Fail");
                }
            } else {
                console.log('No user found with the provided username.');
                res.status(404).send('User not found');
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
            return res.status(409).send('Username already exists');
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

                    return res.status(201).send('User registered successfully');
                }
            );
        }
    });
});




const port = 3000;
app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
});