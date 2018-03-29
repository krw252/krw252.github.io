---
layout: post
title: Classifacoustics
---

If there is one thing I look forward to on a Monday, its got to be my morning commute.  Nothing better than waking up early to pack myself into a just-too-warm-and-somehow-humid train for an hour. Well maybe not, but it is usually the first time I listen to my new Spotify *Discover Weekly* playlist and boy, it never fails me.  

Being someone who listens to a *very* wide range of genres, I am wholly impressed in how Spotify's recommendation engines are so good at picking up nuances in my recent music tastes. It feels like I have a soundtrack curated for me for the week ahead.  And its not like its just throwing me tracks or artists I already listen to, its providing me with actually providing unearthing discoveries.  Neat.

I wanted to learn more about this wizardry.  I knew Spotify had a very detailed set of [genres](http://everynoise.com/engenremap.html) but how?  What does the acoustic data from [Echo Nest](https://techcrunch.com/2014/03/06/spotify-acquires-the-echo-nest/) look like?  And most importantly, could I build something like this myself with machine learning classification (supervised) and clustering (unsupervised) techniques?  We'll see...

##### Sifting through the Raw Raw
To get my dataset, I leveraged Spotify Web API through a nice little Python library called [Spotipy](https://github.com/plamere/spotipy).  They have excellent documentation, I'd recommend checking it out.  The first thing you have to do is [get your project authenticated.](https://developer.spotify.com/web-api/authorization-guide/)  Once setup, you can pull a ton of [information from Spotify](http://spotipy.readthedocs.io/en/latest/#api-reference).  For the purposes of this project, I wanted to look at all raw acoustic features as well as some track metadata to get an idea of how the raw features might affect song popularity.  To do so, I wanted to pull a random set of songs across a range of pre-defined genres.  I went with broad genres with as much musical variety as possible - after testing a few sets I went with these:

```python
genres_narrow = ['blues','classical','country','techno', 'trance', 'house',
                 'folk','jazz','ambient','reggae','rock', 'punk', 'metal',
                 'hip hop', 'soundtrack', 'holiday', 'acoustic', 'rnb',
                'funk','disney', 'pop']
```

Finally, I built out some functions to pull the metadata and acoustic features I wanted.  As these two searches were pulled separately, I had to join my dataframes after by searching for track-id's.  Note how I randomized my dataset by creating a random offset for each iteration of my search. Below is an example of the code I used:

```python
# n is the number of searches per genre - grabs 50 songs per search
# keep n reasonable for rate-limiting purposes
# offset randomizes your sample, off is an int between 0-100000
def get_all(n,off,list_of_genres):
    # helper function to get meta data
    def get_meta(name,off):
        # max limit is 50
        results = sp.search(q=name,limit=50,offset=off)
        temp = defaultdict(list)
        for t in results['tracks']['items']:
            temp['track'].append(t['name'])
            temp['tid'].append(t['id'])
            temp['artist'].append(t['artists'][0]['name'])
            temp['aid'].append(t['artists'][0]['id'])
            temp['tpopularity'].append(t['popularity'])
            temp['explicit'].append(t['explicit'])
            temp['duration_min'].append(round(t['duration_ms']/1000/60,2))
            temp['search_term'].append(name)
        df = pd.DataFrame(temp)
        return df

    # helper function to get acoustic features
    def get_features(trackID):
        features = sp.audio_features(trackID)
        temp = defaultdict(list)
        for song in features:
            for k, v in song.items():
                temp[k].append(v)
        df = pd.DataFrame(temp)
        df.rename(columns={'id': 'tid'}, inplace=True)
        df.drop(['analysis_url','track_href','type','uri'], axis=1, inplace=True)
        return df

    # pull meta data
    df_meta = pd.DataFrame(columns=['aid', 'artist', 'duration_min', 'explicit',
                           'search_term', 'tid', 'tpopularity', 'track'])
    for genre in list_of_genres:
        for i in range(0,n):
            offset = random.randint(0,off)
            df_temp = get_meta(genre, offset)
            df_meta = pd.concat([df_meta,df_temp])
    df_meta.drop_duplicates(inplace=True)
    tid = df_meta['tid']
    # pull acoustic features
    df_feat = get_features(tid)
    # join on track id
    df = pd.merge(df_meta, df_feat, on='tid', how='left')
    return df
```
