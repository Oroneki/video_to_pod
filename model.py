from peewee import *

db = SqliteDatabase('podcasts.db')


class Podcast(Model):
    nome = CharField()
    youtube_id = CharField(unique=True, index=True)
    data = DateField()
    baixado = BooleanField()
    baixado_em = DateTimeField(null=True)
    fase = IntegerField()
    arquivo_baixado = CharField(null=True)
    arquivo_podcast = CharField(null=True)
    duracao = CharField(null=True)
    extra1 = CharField(null=True, db_column="ext_1")
    extra2 = CharField(null=True, db_column="ext_2")
    extra3 = CharField(null=True, db_column="ext_3")
    extra4 = CharField(null=True, db_column="ext_4")
    extra5 = CharField(null=True, db_column="ext_5")
    extra6 = CharField(null=True, db_column="ext_6")
    extra7 = CharField(null=True, db_column="ext_7")
    extra8 = CharField(null=True, db_column="ext_8")
    extra9 = CharField(null=True, db_column="ext_9")

    class Meta:
        database = db


db.create_tables([Podcast], safe=True)
