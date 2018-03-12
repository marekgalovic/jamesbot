from peewee import MySQLDatabase, Model, PrimaryKeyField, DecimalField, CharField, DateField, IntegerField, BooleanField
from dateutil.parser import parse
import json
import re

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

def parse_duration(value):
    matches = re.search('(\d+)', str(value))
    if matches is not None:
        return int(matches.group(0))

def get_condition(field, value):
    if value is None:
        return None
    
    if field == 'or_city':
        return Trips.or_city ** '%{0}%'.format(value)
    elif field == 'dst_city':
        return Trips.dst_city ** '%{0}%'.format(value)
    elif field == 'str_date':
        return Trips.str_date >= parse(value)
    elif field == 'end_date':
        return Trips.end_date <= parse(value)
    elif field == 'budget':
        return Trips.price <= float(values)
    elif field == 'seat':
        return Trips.seat == value
    elif field == 'max_duration':
        return Trips.duration <= parse_duration(value)
    elif field == 'min_duration':
        return Trips.duration >= parse_duration(value)

    return None

def query(frame):
    conditions = []
    for (field, value) in frame.items():
        cond = get_condition(field, value)
        if cond is not None:
            conditions.append(cond)
    
    if len(conditions) == 0:
        return []

    db_cond = conditions[0]
    if len(conditions) > 1:
        for condition in conditions:
            db_cond &= condition
    
    result = []
    for trip in Trips.select().where(db_cond).limit(10).dicts():
        result.append(trip)

    return result
