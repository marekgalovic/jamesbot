from peewee import MySQLDatabase, Model, PrimaryKeyField, DecimalField, CharField, DateField, IntegerField, BooleanField
from dateutil.parser import parse
import json
from optparse import OptionParser

# from . import Trips

parser = OptionParser()
parser.add_option('--db-file', dest='db_file')
options, _ = parser.parse_args()

print('DB:', options.db_file)

db = MySQLDatabase(database='jamesbot', host='127.0.0.1', port=3406, user='root')

class Trips(Model):
    class Meta:
        database = db

    id = PrimaryKeyField()
    name = CharField()
    price = DecimalField()
    or_city = CharField()
    dst_city = CharField()
    str_date = DateField()
    end_date = DateField()
    duration = IntegerField(null=True)
    category = CharField(null=True)
    seat = CharField(null=True)
    gst_rating = DecimalField(null=True)
    wifi = BooleanField(null=True)
    parking = BooleanField(null=True)
    breakfast = BooleanField(null=True)
    spa = BooleanField(null=True)

with open(options.db_file, 'r') as f:
    db_items = json.load(f)

with db.atomic():
    for db_item in db_items:
        db_item['str_date'] = parse(db_item['str_date'])
        db_item['end_date'] = parse(db_item['end_date'])
        Trips.insert(db_item).execute()
    # Trips.insert_many(db_items).execute()


# for trip in Trips.select():
#     print(trip)
