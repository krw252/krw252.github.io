---
layout: post
title: Classifacoustics
---

If there is one thing I look forward to on a Monday, it must be the morning commute.  Nothing beats a just-too-warm-and-somehow-humid hour long train ride. Well maybe not, but it is usually the first chance I get to listen to my new Spotify *Discover Weekly* playlist and boy, it never fails me.  

Being someone who listens to a *very* wide range of genres, I am wholly impressed in how Spotify's recommendation engines are so good at picking up nuances in my recent music tastes. It feels like I have a soundtrack curated for me for the week ahead.  And its not like its just throwing me tracks or artists I already listen to, its providing me with actually providing unearthing discoveries.  Neat.

I wanted to learn more.  I knew Spotify had a very detailed set of [genres](http://everynoise.com/engenremap.html) but how are these labeled?  And more importantly, could I build a simple recommender myself?  To do so I decided to use the [Echo Nest](https://techcrunch.com/2014/03/06/spotify-acquires-the-echo-nest/) acoustic data to:

1. Run a tree based classifier to label track genres and...

2. Build a simple distance based recommendation engine.

##### Sifting through the Raw Raw
To get my dataset, I leveraged Spotify Web API through a nice little Python library called [Spotipy](https://github.com/plamere/spotipy).  They have excellent documentation, I'd recommend checking it out.  The first thing you have to do is [get your project authenticated.](https://developer.spotify.com/web-api/authorization-guide/)  Once setup, you can pull a ton of [information from Spotify](http://spotipy.readthedocs.io/en/latest/#api-reference).  For this analysis, I extracted track metadata (title, artist, duration, genre from search term) and acoustic features for over 30k tracks.  As you can see from the distribution below, the genres are pretty uneven.  This is fine when we make our recommender system later but will need to be resampled for our classification task.

![distribution]({{site.url}}/images/distribution.jpg)

Here is a snippet of code to upsample each genre to 2,000 records each.  You can see the final feature set I settled on after iterating through a few sets:
```python
def upsample(n,categories):
    df = pd.DataFrame(columns=['acousticness', 'danceability', 'energy', 'instrumentalness',
                                'key','liveness', 'loudness', 'speechiness', 'tempo',
                                'time_signature', 'valence', 'duration_min', 'explicit',
                                'mode', 'tid', 'aid', 'artist','track', 'genre', 'pop'])
    for cat in categories:
        c = train_df[train_df['genre'] == cat]
        if len(c) < n:
            temp = resample(c, replace=True, n_samples=(n), random_state=123)
            df = pd.concat([df,temp])
        elif len(c) > (n * 1.1):
            temp = resample(c, replace=False, n_samples=(n), random_state=123)
            df = pd.concat([df,temp])
        else:
            df = pd.concat([df,c])
    return df
```

Now that we have even samples, let's see if we how accurately we can classify these tracks based on their acoustic features.  Something to be wary of at this juncture is scale and normalization, particularly if you are using something like an SVM or K-means with a Euclidean distance function.  Since We are leveraging a Random Forest below we don't need to worry about this.

For my final feature set, I chose the below after iterating through a few variations.

```python
['acousticness', 'danceability', 'energy',
 'instrumentalness', 'key', 'liveness', 'loudness',
  'speechiness', 'tempo', 'time_signature',
  'valence', 'duration_min', 'explicit', 'mode', 'tid',
  'aid', 'artist', 'track', 'genre', 'pop']
```
##### Random forest

|metric|result|
|------|------|
|accuracy| 0.31|
|precision| 0.29|
|recall| 0.30|

![confusion_matrix]({{site.url}}/images/rf_cm.jpg)
