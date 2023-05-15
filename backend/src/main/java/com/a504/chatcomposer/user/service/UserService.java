package com.a504.chatcomposer.user.service;

import com.a504.chatcomposer.music.dto.enums.Genre;
import com.a504.chatcomposer.user.dto.response.UserResp;
import com.a504.chatcomposer.user.entity.FavoriteGenre;
import com.a504.chatcomposer.user.entity.User;
import com.a504.chatcomposer.user.repository.FavoriteGenreRepository;
import com.a504.chatcomposer.user.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

@Service
@RequiredArgsConstructor
public class UserService {
    private final UserRepository userRepository;
    private final FavoriteGenreRepository favoriteGenreRepository;

    public User getUser(String userId) {
        return userRepository.findByUserId(userId);
    }

    public Long getUserSeq(String userId) {
        return userRepository.findByUserId(userId).getUserSeq();
    }

    public List<Genre> getUserGenre(String userId) {
        List<FavoriteGenre> favoriteGenreList = favoriteGenreRepository.findByUser_UserId(userId);
        List<Genre> genreList = new ArrayList<>();
        for (FavoriteGenre g : favoriteGenreList) {
            genreList.add(g.getGenre());
        }
        return genreList;
    }

    public UserResp getUserResp(String userId) {
        User user = this.getUser(userId);
        List<Genre> genreList = this.getUserGenre(userId);

        return new UserResp(user, genreList);
    }

}