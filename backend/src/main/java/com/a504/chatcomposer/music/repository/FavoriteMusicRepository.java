package com.a504.chatcomposer.music.repository;

import java.util.Optional;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import com.a504.chatcomposer.user.entity.FavoriteMusic;

@Repository
public interface FavoriteMusicRepository extends JpaRepository<FavoriteMusic, Long> {
	
	Optional<FavoriteMusic> findByMusic_idAndUser_UserSeq(Long musicId, Long loginUserId);
}
