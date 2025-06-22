import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email(subject, body, to_emails):
    # SMTP server configuration
    sender_email = "2785242003@qq.com"
    password = "vmhptddjegmndfcj"
    smtp_server = 'smtp.qq.com'
    smtp_port = 587
    smtp_user = sender_email
    smtp_password = password

    # Create the email
    msg = MIMEMultipart()
    msg['From'] = smtp_user
    msg['To'] = ', '.join(to_emails)
    msg['Subject'] = subject

    # Attach the email body
    msg.attach(MIMEText(body, 'plain'))

    try:
        # Connect to the SMTP server
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # Secure the connection
        server.login(smtp_user, smtp_password)  # Login to the SMTP server
        server.sendmail(smtp_user, to_emails, msg.as_string())  # Send the email
        server.quit()  # Close the connection
        print('Email sent successfully')
    except Exception as e:
        print(f'Failed to send email: {e}')