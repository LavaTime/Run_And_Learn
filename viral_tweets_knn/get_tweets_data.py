from flask_wtf import FlaskForm
from wtforms.fields import StringField, SubmitField, SelectField, IntegerField, DateField
from datetime import datetime

from wtforms.validators import NumberRange, DataRequired


class get_tweets_data(FlaskForm):
    tweet_length = IntegerField(
        'tweet length',
        default='100',
        validators=[DataRequired(), NumberRange(min=1, max=280)],
        render_kw={'class': 'inputs'}
    )
    followers_count = IntegerField(
        'followers count',
        default='1200',
        validators=[DataRequired(), NumberRange(min=0)],
        render_kw={'class': 'inputs'}
    )
    friends_count = IntegerField(
        'friends count',
        default='1400',
        validators=[DataRequired(), NumberRange(min=0)],
        render_kw={'class': 'inputs'}
    )
    submit = SubmitField('submit',
                         render_kw={'class': 'inputs'})
