import base64
import getpass
from os import environ
from Crypto.Protocol.KDF import scrypt
from Crypto.Cipher import AES

global password
password = None

PREFIX = 'ENCRYPT{'
SUFFIX = '}'
SCRYPT_PARAM_N = 2**16
SCRYPT_PARAM_R = 8

def _get_password() -> str:
    global password
    if password is None:
        if 'SECRET_PASSWORD' in environ:
            password = environ['SECRET_PASSWORD']
        else:
            password = getpass.getpass(prompt='Enter your password $ ')
    return password

def _get_secure_value(value: str):
    if value.startswith(PREFIX) and value.endswith(SUFFIX):
        password = _get_password()

        nonce, ciphertext, tag = [base64.b64decode(x) for x in value[len(PREFIX):-len(SUFFIX)].split(';')]
        key = scrypt(password, '', 32, N=SCRYPT_PARAM_N, r=SCRYPT_PARAM_R, p=1)

        cipher = AES.new(key, AES.MODE_EAX, nonce=nonce)
        plaintext = cipher.decrypt(ciphertext)
        try:
            cipher.verify(tag)
            return plaintext.decode('utf-8')
        except ValueError:
            print('Incorrect password')
            raise Exception("Invalid Password Exception")
    return value

def secure_config(config: dict) -> dict:
    n_config = {}
    for key, value in config.items():
        if isinstance(value, str):
            n_config[key] = _get_secure_value(value)
        elif isinstance(value, dict):
            n_config[key] = secure_config(value)
        else:
            n_config[key] = value

    return n_config

def enc_secure_config(value: str):
    password = _get_password()

    # No need for salt, the key is always used with nonce
    key = scrypt(password, '', 32, N=SCRYPT_PARAM_N, r=SCRYPT_PARAM_R, p=1)
    
    cipher = AES.new(key, AES.MODE_EAX)
    nonce = cipher.nonce
    ciphertext, tag = cipher.encrypt_and_digest(value.encode('utf-8'))

    value = ';'.join([base64.b64encode(x).decode('utf-8') for x in [nonce, ciphertext, tag]])
    return PREFIX + value + SUFFIX