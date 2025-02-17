const nodemailer = require('nodemailer');

// Create reusable transporter
const transporter = nodemailer.createTransport({
    host: process.env.SMTP_HOST || 'smtp.gmail.com',
    port: process.env.SMTP_PORT || 587,
    secure: false, // true for 465, false for other ports
    auth: {
        user: process.env.SMTP_USER,
        pass: process.env.SMTP_PASS
    }
});

async function sendResultsEmail(email, resultId) {
    const resultsUrl = `${process.env.BASE_URL || 'http://localhost'}/results.html?resultId=${resultId}`;
    
    try {
        await transporter.sendMail({
            from: process.env.SMTP_FROM || '"CRISPR Guide Design" <no-reply@example.com>',
            to: email,
            subject: 'Your CRISPR Guide Results Are Ready',
            text: `Your guide design results are ready! View them at: ${resultsUrl}`,
            html: `
                <h2>Your CRISPR Guide Results Are Ready</h2>
                <p>Your guide design job has completed. Click the link below to view your results:</p>
                <p><a href="${resultsUrl}">${resultsUrl}</a></p>
            `
        });
        console.log(`Results email sent to ${email}`);
    } catch (error) {
        console.error('Error sending email:', error);
        throw error;
    }
}

module.exports = { sendResultsEmail }; 