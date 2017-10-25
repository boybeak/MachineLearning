class Movie():
    
    def __init__(self, title, poster_image_url, trailer_youtube_url):
        """
        this function is contructor function for initializing the Move class instance
        """
        self.title = title
        self.poster_image_url = poster_image_url
        self.trailer_youtube_url = trailer_youtube_url
        print ("the movie info as below title=" + self.title + " poster_image=" + self.poster_image_url + " trailer_youtube_url=" + self.trailer_youtube_url)
