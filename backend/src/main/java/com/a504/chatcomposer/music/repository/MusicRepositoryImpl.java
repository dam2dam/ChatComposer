package com.a504.chatcomposer.music.repository;

import static com.a504.chatcomposer.global.util.Utils.*;

import java.util.List;

import com.a504.chatcomposer.user.entity.QUser;
import org.springframework.util.StringUtils;

import com.a504.chatcomposer.global.exception.CustomException;
import com.a504.chatcomposer.global.exception.CustomExceptionType;
import com.a504.chatcomposer.user.entity.QFavoriteMusic;
import com.a504.chatcomposer.music.dto.enums.Genre;
import com.a504.chatcomposer.music.dto.response.MusicDetailResp;
import com.a504.chatcomposer.music.dto.response.MusicsResp;
import com.a504.chatcomposer.music.dto.response.QMusicDetailResp;
import com.a504.chatcomposer.music.dto.response.QMusicsResp;
import com.a504.chatcomposer.music.entity.Music;
import com.a504.chatcomposer.music.entity.QMusic;
import com.a504.chatcomposer.music.entity.QMusicTag;
import com.a504.chatcomposer.music.entity.QPrompt;
import com.a504.chatcomposer.music.entity.QTrack;
import com.a504.chatcomposer.tag.entity.QTag;
import com.querydsl.core.types.dsl.BooleanExpression;
import com.querydsl.core.types.dsl.Expressions;
import com.querydsl.jpa.impl.JPAQueryFactory;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

/**
 * Music 관련 디비 쿼리를 구현한 클래스 (Querydsl 사용)
 */
@RequiredArgsConstructor
@Slf4j
public class MusicRepositoryImpl implements MusicCustomRepository {

	private final JPAQueryFactory jpaQueryFactory;
	QFavoriteMusic favoriteMusic = QFavoriteMusic.favoriteMusic;
	QMusic music = QMusic.music;
	QMusicTag musicTag = QMusicTag.musicTag;
	QTag tag = QTag.tag;
	QTrack track = QTrack.track;
	QPrompt prompt = QPrompt.prompt;
	QUser user = QUser.user;

	@Override
	public List<MusicsResp> getMusicList(Integer genre, String tagName, String nickname, String title,
		String isMyFavorite,
		Long loginUserId) {

		log.info("MusicRepositoryImpl | getMusicList() loginUserId : {} ", loginUserId);

		List<MusicsResp> musicsResps = jpaQueryFactory
			.select(new QMusicsResp(
				Expressions.asNumber(isMember(loginUserId)).as("loginUserId"),
				music,
				music.user.userSeq,
				music.user.nickname)).distinct()
			.from(music)
			.leftJoin(music.musicTags, musicTag)
			.leftJoin(musicTag.tag, tag)
			.leftJoin(music.favoriteMusics, favoriteMusic)
			.where(
				genreEq(genre),
				tagNameContains(tagName),
				nicknameContains(nickname),
				titleContains(title),
				// TODO: '나'의 선호음악이므로 로그인 회원 정보를 사용해야 함
				isLikedEq(isMyFavorite, loginUserId)
			).fetch();

		log.info("MusicRepositoryImpl | getMusicList() musicsResps : {}", musicsResps);

		// return !CollectionUtils.isEmpty(musicsResps) ? musicsResps : Collections.emptyList();
		return musicsResps;
	}

	@Override
	public MusicDetailResp getMusicDetail(Long musicId, Long loginUserId) {

		log.info("MusicRepositoryImpl | getMusicDetail() loginUserId : {} ", loginUserId);

		Music existMusic = jpaQueryFactory
			.selectFrom(music)
			.where(
				music.id.eq(musicId)
			)
			.fetchOne();
		if (existMusic == null) {
			throw new CustomException(CustomExceptionType.MUSIC_NOT_FOUND);
		}

		MusicDetailResp musicDetailResp = jpaQueryFactory
			.select(new QMusicDetailResp(
				Expressions.asNumber(isMember(loginUserId)).as("loginUserId"),
				music,
				user.userSeq,
				user.nickname
			))
			.from(music)
			.leftJoin(music.user, user)
//			.leftJoin(member.memberProfile, memberProfile)
			.leftJoin(music.tracks, track)
			.leftJoin(track.prompt, prompt)
			.leftJoin(music.musicTags, musicTag)
			.join(musicTag.tag, tag)
			.where(
				music.id.eq(musicId)
			)
			.fetchFirst();

		log.info("MusicRepositoryImpl | getMusicDetail() musicDetailResp : {} ", musicDetailResp);

		return musicDetailResp;
	}

	/**
	 * @return loginUserId (Login), or -1L (Not login)
	 */
	private Long isMember(Long loginUserId) {
		return loginUserId != null ? loginUserId : NOT_LOGIN;
	}

	/**
	 * @return music.genre=?, or null
	 */
	private BooleanExpression genreEq(Integer genre) {
		return genre != null ? music.genre.eq(Genre.findByNumber(genre)) : null;
	}

	/**
	 * @return tag.tag_name like ? escape '!', or null
	 */
	private BooleanExpression tagNameContains(String tagName) {
		return tagName != null ? tag.tagName.contains(tagName) : null;
	}

	/**
	 * @return member_profile.nickname like ? escape '!', or null
	 */
	private BooleanExpression nicknameContains(String nickname) {
		return StringUtils.hasText(nickname) ? music.user.nickname.contains(nickname) : null;
	}

	/**
	 * @return music.title like ? escape '!', or null
	 */
	private BooleanExpression titleContains(String title) {
		return StringUtils.hasText(title) ? music.title.contains(title) : null;
	}

	/**
	 * @return favorite_music.member_id=?, or null
	 */
	private BooleanExpression isLikedEq(String isMyFavorite, Long loginUserId) {
		return StringUtils.hasText(isMyFavorite) && isMyFavorite.equals(YES) ? favoriteMusic.user.userSeq.eq(loginUserId) :
			null;
	}
}
