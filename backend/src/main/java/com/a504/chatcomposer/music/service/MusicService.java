package com.a504.chatcomposer.music.service;

import java.util.List;

import org.springframework.stereotype.Service;

import com.a504.chatcomposer.music.dto.response.MusicDetailResp;
import com.a504.chatcomposer.music.dto.response.MusicsResp;
import com.a504.chatcomposer.music.repository.MusicRepository;

import lombok.RequiredArgsConstructor;

@Service
@RequiredArgsConstructor
public class MusicService {

	private final MusicRepository musicRepository;

	/**
	 * 음악 리스트 조회
	 */
	public List<MusicsResp> getMusicList(Integer genre, String tag, String nickname, String title, String isMyFavorite,
		Long loginUserId) {

		return musicRepository.getMusicList(genre, tag, nickname, title, isMyFavorite, loginUserId);
	}

	/**
	 * 음악 상세 조회
	 */
	public MusicDetailResp getMusicDetail(Long musicId, Long loginUserId) {

		return musicRepository.getMusicDetail(musicId, loginUserId);
	}
}
