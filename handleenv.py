import os

def handleenv(env=".env"):
    with open(env, encoding="utf-8") as f:
        env_lines = f.readlines()        
    for line in env_lines:
        line=line.strip()
        if line.startswith('#'): continue
        if len(line) <= 1: continue
        l = line.split('=')
        if len(l) != 2:
            print('!> VARIAVEL DE AMBIENTE MAL RESOLVIDA!', line)
            print(len(line))
            continue
        print(str(l[0]), str(l[1]), sep=" -> ")
        os.environ.setdefault(str(l[0]), str(l[1]))


if __name__ == '__main__':
    handleenv()