package com.a504.chatcomposer.user.service;

import com.a504.chatcomposer.music.dto.enums.Genre;
import com.a504.chatcomposer.user.dto.request.UserUpdateReq;
import com.a504.chatcomposer.user.dto.response.UserResp;
import com.a504.chatcomposer.user.entity.FavoriteGenre;
import com.a504.chatcomposer.user.entity.User;
import com.a504.chatcomposer.user.repository.FavoriteGenreRepository;
import com.a504.chatcomposer.user.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayList;
import java.util.Collections;
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

    @Transactional
    public void updateUser(String userId, UserUpdateReq userUpdateReq) {
        User user = this.getUser(userId);
        if(!user.getNickname().equals(userUpdateReq.getNickname())){ //닉네임이 기존과 다르면
            user.setNickname(userUpdateReq.getNickname());
            userRepository.save(user);
        }

        List<Genre> genreCurrentList = this.getUserGenre(userId); //현재 장르리스트
        List<Integer> genreUpdateList = userUpdateReq.getFavoriteGenres(); //수정할 장르 리스트 (DB에 반영될 리스트)

        //삭제할 장르 리스트 구하기
        List<Genre> genreDeleteList = new ArrayList<>(); //삭제할 장르 : 업데이트 리스트에 없는 장르
        for (Genre g : genreCurrentList) {
            if(!genreUpdateList.contains(g.getNumber())){ //수정할 리스트에 현재 장르가 없으면
                genreDeleteList.add(g); //삭제할 장르 리스트에 추가
            }
        }
        //장르 삭제
        for (Genre g : genreDeleteList) {
            favoriteGenreRepository.deleteFavoriteGenreByUser_UserIdAndGenre(userId, g);
        }


        //추가할 장르 리스트 구하기
        List<Genre> genreSaveList = new ArrayList<>(); //추가할 장르 : 현재 리스트에 없는 장르
        for (Integer g : genreUpdateList) {
            if (!genreCurrentList.contains(Genre.findByNumber(g))) { //현재 리스트에 수정할 장르가 없으면
                genreSaveList.add(Genre.findByNumber(g)); //추가할 장르 리스트에 추가
            }
        }
        //장르 추가
        for (Genre g : genreSaveList) {
            FavoriteGenre favoriteGenre = FavoriteGenre.builder()
                    .genre(g)
                    .user(user)
                    .build();
            favoriteGenreRepository.save(favoriteGenre);
        }


    }
}