export PYTHONPATH=$PYTHONPATH:`pwd`/flaskr:PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/../models/research/slim:`pwd`/../models/research/
export FLASK_ENV=development
export FLASK_APP=.
export BUCKET=imagerie1
export ACCESS_KEY=imagerie
export SECRET_KEY=imagerie
export S3_URL=http://localhost:9000
export REDIS_SERVER=localhost
export REDIS_PORT=6379