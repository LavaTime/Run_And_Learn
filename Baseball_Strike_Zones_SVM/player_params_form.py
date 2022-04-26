from flask_wtf import FlaskForm
from wtforms.fields import StringField, SubmitField, SelectField, IntegerField, DateField
from datetime import datetime

from wtforms.validators import DataRequired


class player_params_form(FlaskForm):
    player_name = StringField(
        'player name',
        default='Chris Sale',
        validators=[DataRequired()],
        render_kw={'class': 'inputs'}
    )
    player_id = IntegerField('Player ID')
    start_date = DateField(
        'start date',
        default=datetime(2008, 4, 1),
        validators=[DataRequired()],
        render_kw={'class': 'inputs'}
    )
    end_date = DateField(
        'end date',
        default=datetime(2017, 7, 15),
        validators=[DataRequired()],
        render_kw={'class': 'inputs'}
    )
    submit = SubmitField('submit',
                         render_kw={'class': 'inputs'})
