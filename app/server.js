const express = require('express');
const app = express();
const path = require('path');
const port = 3000;
const nodemailer = require('nodemailer');
require('dotenv').config();
const bodyParser = require('body-parser');

app.use(express.static(__dirname + '/public'));
app.use(bodyParser.json())
app.use(bodyParser.urlencoded({ extended: false }))


app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname + '/public/home.html'));
});

app.post('/send-email', (req, res) => {
    sendEmail(req.body.email, req.body.subject, req.body.text)
        .then(() => {
            res.status(201).json('Email sent!')
        })
        .catch((error) => {
            console.log(error);
        });
});

app.listen(port, () => {
    console.log(`App listening at http://localhost:${port}`);
});

const sourceEmailUser = {
    user: process.env.EMAIL,
    pass: process.env.PASSWORD
}

const sendEmail = (email, subject, text) => {
    let transporter = nodemailer.createTransport({
        service: 'gmail',
        auth: sourceEmailUser
    });

    let mailOptions = {
        from: sourceEmailUser.user,
        to: email,
        subject: subject,
        html: `<h3>Dear ${email.split('@')[0]},</h3>
        <p>Thank you for completing the Mental Health Survey. Based on our analysis, we have some insights to share with you.</p>
        <p>${text}</p>
        <p>Thank you for your time and we hope you have a great day!</p>
        <p>Best Regards,</p>
        <p>Group 6</p>`
    };
    
    return new Promise((resolve, reject) => {
        transporter.sendMail(mailOptions, function (error, data) {
            if (error) {
                reject(error);
            }
            resolve(data);
        })
    });
}