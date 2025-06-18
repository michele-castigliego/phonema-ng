from espeakng import ESpeakNG

esng = ESpeakNG()
esng.voice = 'it'  # Imposta la voce italiana
print(esng.g2p("ciao", ipa=True))

