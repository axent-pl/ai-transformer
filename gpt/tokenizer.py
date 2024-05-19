import json
import os
import subprocess
from typing import List

class Tokenizer:

    def __init__(self, bin_dir_path: str, params_path: str) -> None:
        assert os.path.exists(bin_dir_path)
        assert os.path.exists(os.path.join(bin_dir_path,'encode'))
        assert os.path.exists(os.path.join(bin_dir_path,'decode'))
        assert os.path.exists(params_path)
        self.bin_dir_path = bin_dir_path
        self.params_path = os.path.abspath(params_path)
        with open(self.params_path, 'r') as fp:
            self.params = json.load(fp)
            self.vocabulary_size = self.params['last_token']

    def encode(self, text: str) -> List[int]:
        cmd = os.path.join(self.bin_dir_path,'encode')

        result = subprocess.run(
            [cmd,'-params', self.params_path],
            input=text,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise Exception(f"Error: {result.stderr}")
        
        
        return list(map(int,result.stdout.strip('][\n').split(',')))


    def decode(self, tokens: List[int]) -> str:
        cmd = os.path.join(self.bin_dir_path,'decode')
        text = '[{}]'.format(','.join(map(str,tokens)))

        result = subprocess.run(
            [cmd,'-params', self.params_path],
            input=text,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise Exception(f"Error: {result.stderr}")
        
        
        return result.stdout[:-1]