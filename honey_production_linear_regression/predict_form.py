from flask_wtf import FlaskForm
from wtforms.fields import StringField, SubmitField, IntegerField
from wtforms.validators import NumberRange, InputRequired


class PredictForm(FlaskForm):
    start_year = IntegerField(
        'start year',
        default=2013,
        validators=[
            InputRequired('Please enter a start year'),
            NumberRange(min=2013, message='Please enter a a year after 2012')],
        render_kw={'class': 'years'}
    )
    end_year = IntegerField(
        'end year',
        default=2051,
        validators=[
            InputRequired('Please enter an end year'),
            NumberRange(min=2013, message='Please enter a year after 2012')],
        render_kw={'class': 'years'}
    )
    submit = SubmitField(
        'submit',
        render_kw={'class': 'submit'}
    )
