const express = require('express');
const app = express();
const path = require('path');
const port = 3000;
const nodemailer = require('nodemailer');
require('dotenv').config();
const bodyParser = require('body-parser');

/* `app.use(express.static(__dirname + '/public'))` is serving static files from the `public`
directory. This means that any files in the `public` directory can be accessed by the client-side
code using their relative paths. */
app.use(express.static(__dirname + '/public'));
app.use(bodyParser.json())
app.use(bodyParser.urlencoded({ extended: false }))


/* This code block is defining a route for the HTTP GET method at the endpoint '/'. When a GET request
is made to this endpoint, the function is sending the 'home.html' file located in the 'public'
directory using the `res.sendFile()` method. This is essentially serving the home page of the web
application to the user. */
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname + '/public/home.html'));
});

/* This code block is defining a route for the HTTP POST method at the endpoint '/send-email'. When a
POST request is made to this endpoint, the function `sendEmail` is called with the email, subject,
and text parameters extracted from the request body. If the email is successfully sent, a response
with a status code of 201 and a JSON message 'Email sent!' is returned. If there is an error, the
error is logged to the console. */
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

/* `const sourceEmailUser` is an object that contains the email and password of the email account that
will be used to send the email. The email and password are stored as environment variables using the
`dotenv` package, which allows the values to be accessed securely without hardcoding them in the
code. The `user` property of the object contains the email address, and the `pass` property contains
the password. */
const sourceEmailUser = {
    user: process.env.EMAIL,
    pass: process.env.PASSWORD
}

/**
 * The function sends an email using nodemailer with a customized message and subject.
 * @param email - The email address of the recipient to whom the email will be sent.
 * @param subject - The subject of the email that will be sent.
 * @param text - The "text" parameter in the "sendEmail" function is the message body of the email that
 * will be sent to the recipient. It contains the insights based on the analysis of the Mental Health
 * Survey that the recipient completed.
 * @returns A Promise object is being returned.
 */
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
        <p>WellCheck</p>`
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