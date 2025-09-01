import logging
import os
from datetime import datetime

import requests
# from dotenv import load_dotenv
from pytz import timezone

from .timezone import my_timezone

_source = 'USE YOURS'
_target = 'USE YOURS'

log = logging.getLogger()

_fmt = "%Y-%m-%d %H:%M:%S %Z%z"


class EmailSender:

    def __init__(self, exp_id: str, enable: bool):
        self.exp_id = exp_id
        self.enable = enable
        if enable:
            load_dotenv()
            self.MAILGUN_API_KEY = os.getenv('MAILGUN_API_KEY')
            if self.MAILGUN_API_KEY is None:
                log.warning('MAILGUN_API_KEY is not set')
                self.enable = False

    def send(self, subject, content):
        if self.enable:
            subject = str(subject)
            content = str(content)
            try:
                return requests.post(f'https://api.mailgun.net/v3/{_source}/messages',
                                     auth=('api', self.MAILGUN_API_KEY),
                                     data={
                                         'from':
                                         f'<agent name>🤖 <mailgun@{_source}>',
                                         'to': [f'{_target}'],
                                         'subject':
                                         f'[{self.exp_id}] {subject}',
                                         'text':
                                         ('\n\n' + content + '\n\n<your sign off>\n' +
                                          datetime.now(timezone(my_timezone)).strftime(_fmt)),
                                     },
                                     timeout=20)
            except Exception as e:
                log.error(f'Failed to send email: {e}')
