from peewee import *

db = SqliteDatabase('podcasts.db')


class Podcast(Model):
    nome = CharField()
    youtube_id = CharField(unique=True, index=True)
    data = DateField()
    baixado = BooleanField()
    baixado_em = DateTimeField(null=True)
    fase = IntegerField()
    arquivo_baixado = CharField(null = True)
    arquivo_podcast = CharField(null = True)
    duracao = CharField(null = True)
    stats = CharField(null = True)
    extra = CharField(null = True, db_column = "cext")
    extra1 = IntegerField(null = True, db_column = "iext_1")
    extra5 = DateField(null = True, db_column = "dext_5")
    extra2 = CharField(null = True, db_column = "cext_2")
    extra3 = CharField(null = True, db_column = "cext_3")
    extra4 = CharField(null = True, db_column = "cext_4")
    extra6 = DateField(null = True, db_column = "dext_6")
    extra7 = DateField(null = True, db_column = "dext_7")
    extra8 = IntegerField(null = True, db_column = "iext_8")
    extra9 = IntegerField(null = True, db_column = "iext_9")
    extra0 = BooleanField(null = True, db_column = "iext_0")  

    class Meta:
        database = db


db.create_tables([Podcast], safe=True)
