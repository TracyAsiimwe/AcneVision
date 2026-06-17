"""
Database setup and models
"""
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    """User account table."""
    __tablename__ = 'users'

    id         = db.Column(db.Integer, primary_key=True)
    username   = db.Column(db.String(80),  unique=True, nullable=False)
    email      = db.Column(db.String(120), unique=True, nullable=False)
    password   = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationship to analyses
    analyses = db.relationship('Analysis', backref='user', lazy=True)

    def __repr__(self):
        return f'<User {self.username}>'


class Analysis(db.Model):
    """Stores each skin analysis result."""
    __tablename__ = 'analyses'

    id               = db.Column(db.Integer, primary_key=True)
    user_id          = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

    # Analysis results
    severity         = db.Column(db.String(50),  nullable=False)
    confidence       = db.Column(db.Float,        nullable=False)
    health_score     = db.Column(db.Integer,      nullable=False)

    # Feature levels
    blackheads_level     = db.Column(db.String(20))
    whiteheads_level     = db.Column(db.String(20))
    papules_level        = db.Column(db.String(20))
    redness_level        = db.Column(db.String(20))
    hyperpigmentation_level = db.Column(db.String(20))
    texture_level        = db.Column(db.String(20))

    # File paths
    face_image       = db.Column(db.String(200))
    annotated_image  = db.Column(db.String(200))
    heatmap_image    = db.Column(db.String(200))
    report_file      = db.Column(db.String(200))

    # Timestamp
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<Analysis {self.id} - {self.severity}>'

    def to_dict(self):
        return {
            'id'            : self.id,
            'severity'      : self.severity,
            'confidence'    : self.confidence,
            'health_score'  : self.health_score,
            'face_image'    : self.face_image,
            'heatmap_image' : self.heatmap_image,
            'created_at'    : self.created_at.strftime('%B %d, %Y at %I:%M %p'),
        }