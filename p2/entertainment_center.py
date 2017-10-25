import media
import fresh_tomatoes

truman_show = media.Movie(
    "The Truman Show",
    "http://www.gstatic.com/tv/thumb/movieposters/20974/p20974_p_v8_aa.jpg",
    "https://www.youtube.com/watch?v=loTIzXAS7v4")
judgment_day = media.Movie(
    "Terminator 2: Judgment Day",
    "https://upload.wikimedia.org/wikipedia/en/thumb/8/85/Terminator2poster.jpg/220px-Terminator2poster.jpg",
    "https://www.youtube.com/watch?v=7QXDPzx71jQ")
sound_of_music = media.Movie(
    "The Sound of Music",
    "https://upload.wikimedia.org/wikipedia/en/thumb/6/6b/Musical1959-SoundOfMusic-OriginalPoster.png/250px-Musical1959-SoundOfMusic-OriginalPoster.png",
    "https://www.youtube.com/watch?v=lEcKXr3mJ_o")
farewell_my_concubine = media.Movie(
    "Farewell My Concubine",
    "https://upload.wikimedia.org/wikipedia/zh/thumb/0/0c/Bawangbieji.jpg/220px-Bawangbieji.jpg",
    "https://www.youtube.com/watch?v=kfasK2rO0Bw")
devils_on_the_doorstep = media.Movie(
    "Devils On The Doorstep",
    "https://upload.wikimedia.org/wikipedia/zh/thumb/2/21/Devils_on_the_Door_Step.jpg/220px-Devils_on_the_Door_Step.jpg",
    "https://www.youtube.com/watch?v=Qe-jqKBDbxc")
in_the_heat_of_the_sun = media.Movie(
    "In The Heat Of The Sun",
    "https://upload.wikimedia.org/wikipedia/zh/thumb/3/34/%E9%98%B3%E5%85%89%E7%81%BF%E7%83%82.jpg/220px-%E9%98%B3%E5%85%89%E7%81%BF%E7%83%82.jpg",
    "https://www.youtube.com/watch?v=W_BqDM0RhyY")
movie_list = [truman_show, judgment_day, sound_of_music, farewell_my_concubine, devils_on_the_doorstep, in_the_heat_of_the_sun]

fresh_tomatoes.open_movies_page(movie_list)
