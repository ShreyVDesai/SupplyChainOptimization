import smtplib
from email.message import EmailMessage
import pandas as pd

def send_email(emailid, message, subject="Automated Email", 
               smtp_server="smtp.gmail.com", smtp_port=587,
               sender="svarunanusheel@gmail.com", username="svarunanusheel@gmail.com", password="Temp"):
    """
    Sends an email to the given email address.
    
    Parameters:
      emailid (str): Recipient email address.
      message (str, pd.DataFrame, or list): Message content. Can be a plain string, a pandas DataFrame,
                                             or a list containing strings and/or DataFrames.
      subject (str): Subject of the email.
      smtp_server (str): SMTP server address.
      smtp_port (int): SMTP server port.
      sender (str): Sender's email address.
      username (str): Username for SMTP login (if required).
      password (str): Password for SMTP login (if required).
    """
    # Create the EmailMessage object.
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = emailid

    # Determine the type of message and build content accordingly.
    if isinstance(message, str):
        # Simple string message.
        msg.set_content(message)
    elif isinstance(message, pd.DataFrame):
        # Convert DataFrame to text and HTML.
        plain_text = message.to_string()
        html_text = message.to_html()
        msg.set_content(plain_text)
        msg.add_alternative(html_text, subtype='html')
    elif isinstance(message, list):
        # Combine parts that could be strings or DataFrames.
        text_parts = []
        html_parts = []
        for part in message:
            if isinstance(part, str):
                text_parts.append(part)
                html_parts.append(f"<p>{part}</p>")
            elif isinstance(part, pd.DataFrame):
                text_parts.append(part.to_string())
                html_parts.append(part.to_html())
            else:
                # Fallback for other types.
                text_parts.append(str(part))
                html_parts.append(f"<p>{str(part)}</p>")
        combined_text = "\n".join(text_parts)
        combined_html = "".join(html_parts)
        msg.set_content(combined_text)
        msg.add_alternative(combined_html, subtype='html')
    else:
        # Default to string representation for any other type.
        msg.set_content(str(message))
    
    # Sending the email.
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            # Upgrade the connection to secure if using TLS.
            server.starttls()
            if username and password:
                server.login(username, password)
            server.send_message(msg)
        print("Email sent successfully to", emailid)
    except Exception as e:
        print("Failed to send email:", e)

# # Example usage:
# if __name__ == "__main__":
#     # Sending a simple text email.
#     send_email("talksick886@gmail.com", "This is a test email.")

#     # Sending an email with a DataFrame.
#     df = pd.DataFrame({
#         "Name": ["Alice", "Bob"],
#         "Age": [30, 25]
#     })
#     send_email("talksick886@gmail.com", df, subject="DataFrame Email")

#     # Sending an email with a combination of text and DataFrame.
#     message_parts = [
#         "Hello, please see the table below:",
#         df,
#         "Thank you!"
#     ]
#     send_email("talksick886@gmail.com", message_parts, subject="Combined Email")
