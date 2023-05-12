package com.a504.chatcomposer.music.service;

import com.a504.chatcomposer.member.entity.User;
import com.a504.chatcomposer.member.repository.UserRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import com.a504.chatcomposer.global.exception.CustomException;
import com.a504.chatcomposer.global.exception.CustomExceptionType;
import com.a504.chatcomposer.member.entity.FavoriteMusic;
import com.a504.chatcomposer.music.entity.Music;
import com.a504.chatcomposer.music.repository.FavoriteMusicRepository;
import com.a504.chatcomposer.music.repository.MusicRepository;

import lombok.RequiredArgsConstructor;

@Service
@RequiredArgsConstructor
public class FavoriteMusicService {

	private final FavoriteMusicRepository favoriteMusicRepository;
	private final MusicRepository musicRepository;
	private final UserRepository userRepository;

	/**
	 * 음악 좋아요 추가
	 */
	@Transactional
	public void createFavoriteMusic(Long musicId, Long loginUserId) {

		Music music = musicRepository.findById(musicId)
			.orElseThrow(() -> new CustomException(CustomExceptionType.MUSIC_NOT_FOUND));

		User user = userRepository.findByUserSeq(loginUserId)
			.orElseThrow(() ->new CustomException(CustomExceptionType.MEMBER_NOT_FOUND));


		FavoriteMusic favoriteMusic = favoriteMusicRepository.findByMusic_idAndUser_UserSeq(musicId, loginUserId)
			.orElse(null);
		if (favoriteMusic != null) {
			throw new CustomException(CustomExceptionType.DUPLICATE_FAVORITE_MUSIC);
		}

		music.setFavoriteCount(music.getFavoriteCount() + 1);
		favoriteMusicRepository.save(FavoriteMusic.builder()
			.music(music)
			.user(user)
			.build());
	}

	/**
	 * 음악 좋아요 삭제
	 */
	@Transactional
	public void deleteFavoriteMusic(Long musicId, Long loginUserId) {

		Music music = musicRepository.findById(musicId)
			.orElseThrow(() -> new CustomException(CustomExceptionType.MUSIC_NOT_FOUND));

		FavoriteMusic favoriteMusic = favoriteMusicRepository.findByMusic_idAndUser_UserSeq(musicId, loginUserId)
			.orElseThrow(() -> new CustomException(
				CustomExceptionType.FAIL_TO_DELETE_FAVORITE_MUSIC));

		music.setFavoriteCount(music.getFavoriteCount() - 1);
		favoriteMusicRepository.delete(favoriteMusic);
	}
}
