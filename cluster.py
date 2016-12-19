import os
import os.path
from pyspark import SparkContext
from pyspark import SparkConf

def quiet_logs( sc ):
  logger = sc._jvm.org.apache.log4j
  logger.LogManager.getLogger("org").setLevel(logger.Level.ERROR)
  logger.LogManager.getLogger("akka").setLevel(logger.Level.ERROR)

# Setup the context
conf = SparkConf().setMaster("local").setAppName("A3")

# External modules are imported via a separate array. This can also be done
# on a SparkContext that has already been constructed.
sc = SparkContext(conf=conf)
quiet_logs(sc)

print "Loading the dataset..."

datasetPath = 's3://***********/a3/user_artist_data.txt'
artistAliasPath = 's3://***********/a3/artist_alias.txt'
artistDataPath = 's3://***********/a3/artist_data.txt'

rawDataRDD = sc.textFile(datasetPath,2)
rawDataRDD.cache()

print "Loaded AudioScrobbler dataset with {0} points".format(rawDataRDD.count())
head = rawDataRDD.take(10)
print head
#use the whole dataset as the training set
sample=rawDataRDD
print "Let's check whether variables are in range for storing as int32. Here are the statistics for user ID and artist ID:"
print sample.map(lambda x: float(x.split()[0])).stats()
print sample.map(lambda x: float(x.split()[1])).stats()

print "Load the artist ID to name mappings"

rawArtistRDD = sc.textFile(artistDataPath)

def parseArtistIdNamePair(singlePair):
    splitPair = singlePair.rsplit('\t')
    # we should have two items in the list - id and name of the artist.
    if len(splitPair) != 2:
        return []
    else:
        try:
            return [(int(splitPair[0]), splitPair[1])]
        except:
            return []

artistByID = dict(rawArtistRDD.flatMap(lambda x: parseArtistIdNamePair(x)).collect())

print "Artist ID to name mappings loaded"

def parseArtistAlias(alias):
    splitPair = alias.rsplit('\t')
    # we should have two items in the list - id and name of the artist.
    if len(splitPair) != 2:
        #print singlePair
        return []
    else:
        try:
            return [(int(splitPair[0]), int(splitPair[1]))]
        except:
            return []

print "Load artist aliases"
rawAliasRDD = sc.textFile(artistAliasPath)
artistAlias = rawAliasRDD.flatMap(lambda x: parseArtistAlias(x)).collectAsMap()
print "Artist aliases loaded"

print "convert artist IDs to canonical form using aliases"

from pyspark.mllib import recommendation
from pyspark.mllib.recommendation import *
# Let's turn the artist aliases into a broadcast variable.
# That'll distribute it to worker nodes efficiently, so we save bandwidth.
artistAliasBroadcast = sc.broadcast(artistAlias)

def mapSingleObservation(x):
    userID, artistID, count = map(lambda lineItem: int(lineItem), x.split())
    finalArtistID = artistAliasBroadcast.value.get(artistID)
    if finalArtistID is None:
        finalArtistID = artistID
    return Rating(userID, finalArtistID, count)

trainData = sample.map(lambda x: mapSingleObservation(x))
trainData.cache()

# rank = 10
# iterations = 5
# lambda = 0.01
print "Start to build model..."
model = ALS.trainImplicit(trainData, 10, 5, 0.01)
print "Model construction finished..."

#print "Let's see if this model makes sense..."
testUserID = 2093760
print "All artists played by user {0}".format(testUserID)

artistByIDBroadcast = sc.broadcast(artistByID)

artistsForUser = (trainData
                  .filter(lambda observation: observation.user == testUserID)
                  .map(lambda observation: artistByIDBroadcast.value.get(observation.product))
                  .collect())
print artistsForUser

print "Result: ten recommendations for users"

recommendationsForUser = \
    map(lambda observation: artistByID.get(observation.product), model.call("recommendProducts", testUserID, 10))

print recommendationsForUser
