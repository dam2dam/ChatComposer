package com.a504.chatcomposer.user.repository;

import com.a504.chatcomposer.music.dto.enums.Genre;
import com.a504.chatcomposer.user.entity.FavoriteGenre;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface FavoriteGenreRepository extends JpaRepository<FavoriteGenre, Long> {

    List<FavoriteGenre> findByUser_UserId(String userId);

    void deleteFavoriteGenreByUser_UserIdAndGenre(String userId, Genre genre);

}