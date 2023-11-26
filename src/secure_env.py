import base64
import getpass
from Crypto.Protocol.KDF import scrypt
from Crypto.Hash import SHA512
from Crypto.Random import get_random_bytes
from Crypto.Cipher import AES
from Crypto import Random

global password
password = None

PREFIX = 'ENCRYPT{'
SUFFIX = '}'
SCRYPT_PARAM_N = 2**16
SCRYPT_PARAM_R = 8

def secure_config(config: dict) -> dict:
    n_config = {}
    for key, value in config.items():
        if isinstance(value, str):
            n_config[key] = get_secure_value(value)
        elif isinstance(value, dict):
            n_config[key] = secure_config(value)
        else:
            n_config[key] = value

    return n_config

def get_secure_value(value: str):
    if value.startswith(PREFIX) and value.endswith(SUFFIX):
        while True:
            global password
            if password is None:
                password = getpass.getpass(prompt='Enter your password $ ')

            nonce, ciphertext, tag = [base64.b64decode(x) for x in value[len(PREFIX):-len(SUFFIX)].split(';')]
            key = scrypt(password, '', 32, N=SCRYPT_PARAM_N, r=SCRYPT_PARAM_R, p=1)

            cipher = AES.new(key, AES.MODE_EAX, nonce=nonce)
            plaintext = cipher.decrypt(ciphertext).decode('utf-8')
            try:
                cipher.verify(tag)
                return plaintext
            except ValueError:
                print('Incorrect password')
    return value

def enc_secure_config(value: str):
    global password
    if password is None:
        password = getpass.getpass(prompt='Enter your password $ ')

    # No need for salt, the key is always used with nonce
    key = scrypt(password, '', 32, N=SCRYPT_PARAM_N, r=SCRYPT_PARAM_R, p=1)
    
    cipher = AES.new(key, AES.MODE_EAX)
    nonce = cipher.nonce
    ciphertext, tag = cipher.encrypt_and_digest(value.encode('utf-8'))

    value = ';'.join([base64.b64encode(x).decode('utf-8') for x in [nonce, ciphertext, tag]])
    return PREFIX + value + SUFFIX
