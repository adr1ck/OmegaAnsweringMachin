from telethon import TelegramClient
from telethon.sessions import StringSession
from math import log
from re import *
import json
from pyaspeller import YandexSpeller
from googletrans import Translator
from telethon import events
import asyncpg
from os import getenv


api_id = int(getenv('API_ID'))
api_hash = getenv('API_HASH')
bot_token = getenv('BOT_TOKEN')
db_uri = getenv('DB_URI')
chat_id = int(getenv('CHAT_ID'))
observer_session = getenv('OBSERVER_SESSION')

pool: asyncpg.pool.Pool
bot = TelegramClient(StringSession(), api_id, api_hash)
bot.start(bot_token=bot_token)
observer = TelegramClient(StringSession(observer_session), api_id, api_hash)
observer.start()


class Regexes:

    noise_filter = compile(
        r'(@\w+)|'  # @nickname
        r'(\A\S*\s?\S*\Z)|'  # short message
        r'(\A[хапХАПзЗъЪВв\W\d]*\Z)|'  # laughter
        r'(\*{2}[^*+])|'  # forwarded publication
        r'(\w+://\w+)|'  # link
        r'([\d\s\-]{9,})'  # phone number
    )

    time = compile(r'\D([012]?\d\D[0-6]\d)[^\d.]|(сейчас)')

    _days = [
        r'сего[дн]{2}я', 'завтра', 'послезавтра',
        *[r'\W' + d + r'\W' for d in
          ['пн', 'вт', 'ср', 'чт', 'пт', 'сб', 'вс']],
        r'понедельник\w*', r'вторник\w*', r'сред\w+', r'четверг\w*',
        r'пятниц\w+',
        r'суббот\w+', r'воскрисен\w+'
    ]
    date = compile(
        r'\D((([0-3]?\d\D[01]?\d)|'
        r'([01]?\d\D[0-3]?\d))\D(\d{2}|'
        r'\d{4}))\D' +
        r'|(' + ')|('.join(_days) + ')'
    )

    _people = [r'чел\w*', r'поняты\w*', 'людей', 'ребята?']
    _man = [r'[^на]{2}паре?н\w*', r'мальчи[шк]\w*', r'пацан\w*', r'мужчин\w*']
    _woman = [r'девоче?к\w*', r'девуш\w*', r'девчен\w*']
    people = compile('|'.join(_people + _man + _woman))

    num_words = [
        'од(ин(надцат(ь|еро|ерых))?|на|ного)',
        'дв(а|е(надцат(ь|еро|ерых))?|ое|оих|ух)', 'пар(а|у|очку)',
        'тр(и(надцат(ь|еро|ерых))?|ое|ех|оих)',
        'чет(ыр(е|надцат(ь|еро|ерых))|веро|верых)',
        '(пят|шест|сем|восем|девят)(надцат)?(ь|еро|ерых)',
        'десят(ь|еро|ерых)'
    ]
    num = compile(
        r'\D(\d{1,2})\D' +
        '|[^а-я](' + ')[^а-я]|[^а-я]('.join(num_words) + ')[^а-я]'
    )

    forms_dict = {
        ' TIME ': time,
        ' DATE ': date,
        ' NUM ': num,
        ' PEOPLE ': people
    }

    genders_dict = {
        'Мужской': compile('|'.join(_man)),
        'Женский': compile('|'.join(_woman)),
        None: compile('|'.join(_people))
    }


class NBC:
    """
    Naive Bayes classifier for a sequence of events
    """

    def __init__(self, model_data):
        self._model_data = model_data['data']
        self._events = model_data['events']
        self._V = len(self._model_data)
        self._unique_word_data = {
            key: 0 for key in self._events.keys()
        }

    def _laplace(self, C):
        if C:
            return (self._V + self._L('true')) / (self._V + self._L('false'))
        else:
            return 1

    def _D(self, C=None):
        if C is None:
            return sum([i for i in self._events.values()])
        else:
            return self._events[C]

    def _L(self, C):
        return sum([word[C] for word in self._model_data.values()])

    def _Wic(self, i, C):
        return self._model_data.get(i, self._unique_word_data)[C]\
               + self._laplace(C)

    def _P(self, C, Q, Dc, D, Lc, V):
        prior_P = log(Dc / D)
        P_of_events = [log(self._Wic(i, C) / (V + Lc)) for i in Q]
        return prior_P + sum(P_of_events)

    def calculate(self, Q: list, C) -> float:
        return self._P(C, Q, self._D(C), self._D(), self._L(C), self._V)


class User(asyncpg.Record):
    session: str
    switch: bool
    circuit_breaker: bool
    answer: str
    gender: str
    filters = dict()

    replied = False

    def set_attrs(self):
        for var, val in self.items():
            setattr(self, var, val)
        return self

    async def check_authorization(self) -> bool:
        if isinstance(self.session, str) and len(self.session) == 353:
            client = TelegramClient(
                StringSession(self.session),
                api_id,
                api_hash
            )
            await client.connect()
            if await client.is_user_authorized():
                return True
            else:
                await self.authorization_lost()
            await client.disconnect()

        else:
            await self.authorization_lost()
            raise TypeError(
                '"session" must be derived from telethon.StringSession.'
            )

    async def authorization_lost(self):
        await bot.send_message(
            self.id,
            'Ключ авторизации недействителен, '
            'необходима повторная авторизация.\n'
            'Воспользуйтесь: /log_in'
        )
        await pool.execute(
            f'UPDATE users SET session = NULL WHERE id = {self.id}'
        )

    async def reply(self):
        try:
            client = TelegramClient(
                StringSession(self.session),
                api_id,
                api_hash
            )
            await client.connect()
            await client.send_message(chat_id, self.answer)
            self.replied = True
            await client.disconnect()
        except Exception as e:
            await bot.send_message(self.id, str(e))

    async def notify(self, text='Автоответчик сработал!'):
        if self.replied:
            await bot.send_message(self.id, text)

    async def close(self, connection: asyncpg.connection = None):
        if connection is None:
            connection = pool
        if self.replied:
            if self.circuit_breaker:
                await connection.execute(
                    f"UPDATE users SET switch = FALSE WHERE id = {self.id}"
                )
        else:
            await self.check_authorization()
        del self


speller = YandexSpeller()
translator = Translator()
with open('ModelData.json', 'r', encoding='utf-8') as md:
    nbc = NBC(json.load(md))


class ClassifiedMessage:

    def __init__(self, message: events.NewMessage):

        self.original_text = message.text.lower()

        self.processed_text = (
            self.translate() if search(r'[ґєії]', self.original_text)
            else self.spelled()
        ).replace('ъ', 'ь').replace('ё', 'е')

        self.gender = None
        self._determine_gender()

        words = [word for word in split(r'\W', self._descriptor()) if len(word)]

        self.__is_target = all([
            len(words) > 3,
            'NUM' in words or 'PEOPLE' in words,
            nbc.calculate(words, 'true') > nbc.calculate(words, 'false')
        ])

    def __bool__(self):
        if hasattr(self, '_' + self.__class__.__name__ + '__is_target'):
            return self.__is_target

    @property
    def is_target(self) -> bool:
        return self.__is_target

    def translate(self):
        try:
            message = translator.translate(self.original_text, dest='ru')
            if message.src == "uk":
                return message.text
        except Exception as e:
            print('translate:', e)

    def spelled(self):
        try:
            return speller.spelled(self.original_text)
        except Exception as e:
            print('spelled', e)

    def _determine_gender(self):
        genders = [gender
                   for gender, regex in Regexes.genders_dict.items()
                   if search(regex, self.processed_text)]
        if len(genders) == 1 and None not in genders:
            self.gender = genders[0]

    def _descriptor(self):
        text = '  ' + self.processed_text + '  '
        for word, regex in Regexes.forms_dict.items():
            replaced = findall(regex, text)
            while replaced:
                if isinstance(replaced[0], tuple):
                    replaced = [rep for rep in replaced[0] if rep]
                text = text.replace(replaced[0], word, 1)
                replaced = findall(regex, text)
            text = sub(word + r'\s?-\s?' + word, word, text)
        return sub(r'\s+', ' ', text.strip())

    def modifiable_filter(self, filters: list) -> bool:
        filters = [f.lower() for f in filters]
        filters += [
            ' '.join(
                [word for word in split(
                    r'\W', f.replace('ъ', 'ь').replace('ё', 'е')
                ) if len(word)]
            )
            for f in filters
        ]
        text = self.original_text + ' '.join(
            [word for word in split(r'\W', self.processed_text) if len(word)]
        )
        for f in filters:
            if search(f, text):
                return False
        return True


class AnsweringMachine:

    def __init__(self):
        observer.on(
            events.NewMessage(
                chats=[chat_id],
                func=self._noise_filter
            )
        )(self.handler)

    @staticmethod
    def _noise_filter(event):
        return event.text and not search(Regexes.noise_filter,
                                         event.text)

    async def handler(self, event: events.NewMessage):
        message = ClassifiedMessage(event)
        if message:
            connection = await pool.acquire()
            users = await self.get_users(message, connection)
            for user in users:
                await user.reply()
            for user in users:
                await user.notify()
            for user in users:
                await user.close(connection)
            await pool.release(connection)

    @staticmethod
    async def get_users(message: ClassifiedMessage,
                        connection: asyncpg.connection = None) -> list[User, ]:
        if connection is None:
            connection = pool
        gender_par = f" AND gender='{message.gender}'" if message.gender else ""
        users = [u.set_attrs() for u in await connection.fetch(
            "SELECT * FROM users WHERE switch = TRUE AND session IS NOT NULL"
            + gender_par
        )]
        return [user for user in users
                if message.modifiable_filter(user.filters)]


async def create_pool():
    global pool
    pool = await asyncpg.create_pool(
        db_uri,
        record_class=User,
        max_size=5,
        min_size=5
    )


def main():
    bot.loop.run_until_complete(create_pool())
    AnsweringMachine()
    bot.run_until_disconnected()


if __name__ == '__main__':
    main()
