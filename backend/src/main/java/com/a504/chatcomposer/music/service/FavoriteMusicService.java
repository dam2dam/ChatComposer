package com.a504.chatcomposer.music.service;

import org.springframework.stereotype.Service;

import com.a504.chatcomposer.global.exception.CustomException;
import com.a504.chatcomposer.global.exception.CustomExceptionType;
import com.a504.chatcomposer.member.entity.FavoriteMusic;
import com.a504.chatcomposer.member.entity.Member;
import com.a504.chatcomposer.member.repository.MemberRepository;
import com.a504.chatcomposer.music.entity.Music;
import com.a504.chatcomposer.music.repository.FavoriteMusicRepository;
import com.a504.chatcomposer.music.repository.MusicRepository;

import lombok.RequiredArgsConstructor;

@Service
@RequiredArgsConstructor
public class FavoriteMusicService {

	private final FavoriteMusicRepository favoriteMusicRepository;
	private final MusicRepository musicRepository;
	private final MemberRepository memberRepository;

	/**
	 * 음악 좋아요 추가
	 */
	public void createFavoriteMusic(Long musicId, Long loginUserId) {

		Music music = musicRepository.findById(musicId)
			.orElseThrow(() -> new CustomException(CustomExceptionType.MUSIC_NOT_FOUND));

		Member member = memberRepository.findById(loginUserId)
			.orElseThrow(() -> new CustomException(CustomExceptionType.MEMBER_NOT_FOUND));

		music.setFavoriteCount(music.getFavoriteCount() + 1);
		favoriteMusicRepository.save(FavoriteMusic.builder()
			.music(music)
			.member(member)
			.build());
	}
}
